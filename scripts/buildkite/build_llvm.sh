#!/usr/bin/env bash
#
# Script for Buildkite automation only.
# Environment variables must have been declared already.
#
# CMake and build LLVM.

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

LLVMROOT=${HOME}/installs/llvm
mkdir -p ${LLVMROOT}

# LLVM setup
echo "--- LLVM"
LLVM_VERSION=$(llvm_version)
echo "LLVM version: ${LLVM_VERSION}"

# Destination for tar balls
if [ ! "${LLVM_TAR_DIR}" ]; then
  LLVM_TAR_DIR="/tmp/tpp-llvm-tar"
fi
mkdir -p ${LLVM_TAR_DIR}

# Fetch specific LLVM version
LLVM_TAR_NAME=${LLVM_VERSION}.zip
LLVM_TAR_FILE=${LLVM_TAR_DIR}/${LLVM_TAR_NAME}
if [ ! -f "${LLVM_TAR_FILE}" ]; then
  echo_run wget -nv -P ${LLVM_TAR_DIR} https://github.com/llvm/llvm-project/archive/${LLVM_TAR_NAME}
  if [ $? != 0 ]; then
    echo "Failed to fetch repo"
    exit 1
  fi
fi

# Unzip the fetched repo
echo_run unzip -uqn ${LLVM_TAR_FILE} -d ${LLVM_TAR_DIR}
if [ $? != 0 ]; then
  echo "Failed to unpack repo"
  exit 1
fi

# Cleanup tar ball to save device space
rm ${LLVM_TAR_FILE}

LLVM_PROJECT_DIR=${LLVM_TAR_DIR}/llvm-project-${LLVM_VERSION}
LLVM_INSTALL_DIR=${LLVMROOT}/${LLVM_VERSION}

# Environment setup
echo "--- ENVIRONMENT"
if [ ! "${COMPILER}" ]; then
  COMPILER=clang
fi
if [ "${COMPILER}" == "clang" ]; then
  check_program clang
  check_program clang++
  CC=clang
  CXX=clang++
elif [ "${COMPILER}" == "gcc" ]; then
  check_program gcc
  check_program g++
  CC=gcc
  CXX=g++
else
  echo "Compiler "${COMPILER}" not recognized"
  exit 1
fi

if [ ! "${LINKER}" ]; then
  LINKER=lld
fi
check_program ${LINKER}

if [ ! "${LLVM_BUILD_DIR}" ]; then
  LLVM_BUILD_DIR="/tmp/tpp-llvm"
fi
LLVM_BUILD_DIR=$(realpath ${LLVM_BUILD_DIR})
LLVM_BUILD_DIR=${LLVM_BUILD_DIR:-build-${COMPILER}}
mkdir -p ${LLVM_BUILD_DIR}

echo "Environment configured successfully"

 # Configure LLVM
echo "--- CONFIGURE"

LLVM_PROJECTS="mlir"
LLVM_TARGETS="host"
if [ ! "${KIND}" ]; then
  KIND=RelWithDebInfo
fi

echo_run cmake -Wno-dev -G Ninja \
    -B${LLVM_BUILD_DIR} -S${LLVM_PROJECT_DIR}/llvm \
    -DLLVM_ENABLE_PROJECTS=${LLVM_PROJECTS} \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_TARGETS_TO_BUILD=${LLVM_TARGETS} \
    -DCMAKE_BUILD_TYPE=${KIND} \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DLLVM_USE_LINKER=${LINKER} \
    -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}

# Build LLVM
echo "--- BUILD"
echo_run ninja -C ${LLVM_BUILD_DIR} all
if [ $? != 0 ]; then
  exit 1
fi

# Check LLVM
if [ "1" == "${CHECK}" ]; then
  echo "--- CHECK"
  echo_run ninja -C ${LLVM_BUILD_DIR} check-all
  if [ $? != 0 ]; then
    exit 1
  fi
fi

 # Install LLVM
 echo "--- INSTALL"
 mkdir -p ${LLVM_INSTALL_DIR}
 echo_run ninja -C ${LLVM_BUILD_DIR} install
if [ $? != 0 ]; then
  rm -r ${LLVM_INSTALL_DIR}
  exit 1
fi
