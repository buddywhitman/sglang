#!/usr/bin/env bash
# Bring-up script for a brand-new elementary OS 7.x ("Horus", Ubuntu 22.04 jammy base)
# machine, mirroring the WSL2/Ubuntu-24.04 dev box this fork's P-EAGLE work was done on:
# NVIDIA driver + CUDA 13.x, miniforge conda envs (gpu, vllm-test), Slurm+munge+OpenMPI
# single-node "cluster", repo clone, model downloads, and the compute-sanitizer repro.
#
# Usage: sudo ./setup_elementary_os.sh
# Idempotent-ish: safe to re-run; apt/conda/pip steps skip already-satisfied installs.
#
# IMPORTANT: elementary OS 7.x is Ubuntu 22.04 (jammy), NOT 24.04 (noble) like the WSL2
# box this was developed on. NVIDIA's apt repo is keyed by Ubuntu release, so this script
# uses the jammy repo explicitly -- do not copy noble URLs from the WSL2 box.
#
# IMPORTANT: WSL2 uses the Windows host's GPU driver via passthrough (no real Linux
# driver install, no /dev/nvidia* the usual way). Bare metal needs a REAL driver install
# below -- this is not something you can skip or shortcut by looking at the WSL2 box.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run as root: sudo $0" >&2
  exit 1
fi

REAL_USER="${SUDO_USER:-$(logname)}"
REAL_HOME=$(getent passwd "$REAL_USER" | cut -d: -f6)

echo "=== [1/7] apt base packages ==="
apt-get update
apt-get install -y --no-install-recommends \
  build-essential gcc g++ make cmake git curl wget ca-certificates gnupg \
  linux-headers-"$(uname -r)" dkms software-properties-common \
  munge libmunge2 libmunge-dev \
  libopenmpi-dev openmpi-bin openmpi-common \
  libnuma-dev pkg-config

echo "=== [2/7] NVIDIA driver + CUDA toolkit (jammy repo, real bare-metal install) ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  distro=ubuntu2204
  arch=x86_64
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${arch}/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
  dpkg -i /tmp/cuda-keyring.deb
  apt-get update
  # Driver: use the open-kernel-module driver package matching this session's WSL2
  # host driver generation (610.x branch); if unavailable in jammy repos yet, fall
  # back to `ubuntu-drivers autoinstall` for the latest supported driver.
  if apt-cache show nvidia-driver-580-open >/dev/null 2>&1; then
    apt-get install -y nvidia-driver-580-open
  else
    apt-get install -y ubuntu-drivers-common
    ubuntu-drivers autoinstall
  fi
  # CUDA toolkit: match the WSL2 box's 13.3; fall back to whatever cuda-toolkit
  # metapackage jammy's repo currently resolves to if 13-3 isn't published yet.
  if apt-cache show cuda-toolkit-13-3 >/dev/null 2>&1; then
    apt-get install -y cuda-toolkit-13-3
  else
    apt-get install -y cuda-toolkit
  fi
  echo ">>> Driver install requires a REBOOT before nvidia-smi will work."
  echo ">>> Re-run this script after rebooting -- it will skip this step once nvidia-smi succeeds."
else
  echo "nvidia-smi already present, skipping driver/toolkit install:"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
fi

# Make nvcc/cuda visible for this and future shells regardless of exact installed version dir.
CUDA_HOME_DIR=$(ls -d /usr/local/cuda-13.* 2>/dev/null | sort -V | tail -1 || true)
CUDA_HOME_DIR="${CUDA_HOME_DIR:-/usr/local/cuda}"
if [[ -d "$CUDA_HOME_DIR" ]]; then
  cat > /etc/profile.d/cuda.sh <<EOF
export CUDA_HOME=$CUDA_HOME_DIR
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}
EOF
fi

echo "=== [3/7] miniforge (as $REAL_USER) ==="
if [[ ! -d "$REAL_HOME/miniforge3" ]]; then
  sudo -u "$REAL_USER" bash -c "
    curl -4 -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge.sh -b -p $REAL_HOME/miniforge3
  "
fi
source "$REAL_HOME/miniforge3/etc/profile.d/conda.sh"

echo "=== [4/7] conda envs: gpu (sglang), vllm-test (vllm) ==="
# Package set mirrors the WSL2 box: python 3.11, torch 2.11.0+cu130. Editable installs
# of sglang/vllm are done from the cloned repos below (matches how this fork was built,
# not a frozen pip lockfile -- lockfiles for a fast-moving fork rot faster than they help).
for envname in gpu vllm-test; do
  if ! conda env list | grep -q "^${envname} "; then
    sudo -u "$REAL_USER" bash -c "source $REAL_HOME/miniforge3/etc/profile.d/conda.sh && conda create -y -n $envname python=3.11"
  fi
done

sudo -u "$REAL_USER" bash -c "
  source $REAL_HOME/miniforge3/etc/profile.d/conda.sh
  conda activate gpu
  pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0
  pip install triton==3.6.0
"
sudo -u "$REAL_USER" bash -c "
  source $REAL_HOME/miniforge3/etc/profile.d/conda.sh
  conda activate vllm-test
  pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0
"

echo "=== [5/7] Slurm + munge single-node cluster (mirrors WSL2 box's devcluster) ==="
apt-get install -y slurm-wlm slurm-wlm-doc

if [[ ! -s /etc/munge/munge.key ]]; then
  /usr/sbin/mungekey -f
  chown munge:munge /etc/munge/munge.key
  chmod 400 /etc/munge/munge.key
fi
systemctl enable --now munge

HOSTNAME_SHORT=$(hostname -s)
CPU_COUNT=$(nproc)
MEM_MB=$(awk '/MemTotal/{printf "%d", $2/1024}' /proc/meminfo)

mkdir -p /etc/slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
chown slurm:slurm /var/spool/slurmctld /var/log/slurm 2>/dev/null || true

cat > /etc/slurm/slurm.conf <<EOF
ClusterName=devcluster
SlurmctldHost=${HOSTNAME_SHORT}
SlurmctldParameters=enable_config_less
MpiDefault=pmix
SlurmUser=slurm
ProctrackType=proctrack/linuxproc
AuthType=auth/munge
SrunPortRange=60001-60100
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmdPidFile=/var/run/slurmd.pid
SlurmdSpoolDir=/var/spool/slurmd
StateSaveLocation=/var/spool/slurmctld
SwitchType=switch/none
TaskPlugin=task/cgroup
SchedulerType=sched/backfill
SelectType=select/cons_tres
SlurmctldDebug=info
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdDebug=info
SlurmdLogFile=/var/log/slurm/slurmd.log
GresTypes=gpu
NodeName=${HOSTNAME_SHORT} NodeAddr=127.0.0.1 CPUs=${CPU_COUNT} RealMemory=${MEM_MB} Gres=gpu:1 State=UNKNOWN
PartitionName=gpu Nodes=${HOSTNAME_SHORT} Default=YES MaxTime=INFINITE State=UP
EOF

cat > /etc/slurm/gres.conf <<EOF
NodeName=${HOSTNAME_SHORT} Name=gpu File=/dev/nvidia0
EOF

systemctl enable --now slurmctld slurmd

echo "=== [6/7] OpenMPI sanity check ==="
mpicc --version || echo "WARNING: mpicc not found, check libopenmpi-dev install"

echo "=== [7/7] Clone / model download reminders ==="
cat <<'EOF'

Setup script done. Remaining steps (see CLAUDE.md "Elementary OS bring-up" section
for the full checklist and the compute-sanitizer repro command):

  1. If nvidia-smi wasn't available before this run, REBOOT now, then re-run this
     script once to finish the CUDA/toolkit steps it may have skipped.
  2. Clone the repo (if not already):
       git clone https://github.com/buddywhitman/sglang.git ~/contribution/sglang
       cd ~/contribution/sglang && git remote add upstream https://github.com/sgl-project/sglang.git
  3. Editable install into the `gpu` conda env:
       conda activate gpu && pip install -e "python[all]"
  4. Download the Qwen3-1.7B target + EAGLE3 draft pair (see CLAUDE.md for exact
     `hf download` commands and target paths under ~/models/).
  5. Run the P-EAGLE compute-sanitizer repro (CLAUDE.md has the full command) --
     this is the whole point of moving to bare metal: WSL2's WDDM debugger
     interface can't run compute-sanitizer on a consumer GPU, real Linux can.

EOF
