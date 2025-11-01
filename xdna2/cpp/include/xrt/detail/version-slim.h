// SPDX-License-Identifier: Apache-2.0
// Minimal version stub for XRT native build
// Based on libxrt1 2.13.0 and xrt-npu 2.21.0

#ifndef XRT_DETAIL_VERSION_SLIM_H
#define XRT_DETAIL_VERSION_SLIM_H

// Version information for XRT 2.21.0 (NPU) / 2.13.0 (lib)
#define XRT_VERSION_MAJOR 2
#define XRT_VERSION_MINOR 21
#define XRT_VERSION_PATCH 0

// Version code encoding
#define XRT_VERSION(a,b,c) (((a) << 16) + ((b) << 8) + (c))
#define XRT_VERSION_CODE XRT_VERSION(XRT_VERSION_MAJOR, XRT_VERSION_MINOR, XRT_VERSION_PATCH)

// Version extraction macros
#define XRT_MAJOR(code) ((code) >> 16)
#define XRT_MINOR(code) (((code) >> 8) & 0xFF)
#define XRT_PATCH(code) ((code) & 0xFF)

#endif // XRT_DETAIL_VERSION_SLIM_H
