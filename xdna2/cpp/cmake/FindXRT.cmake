# Find XRT package
find_path(XRT_INCLUDE_DIR xrt/xrt_device.h
    PATHS /opt/xilinx/xrt/include
)

find_library(XRT_LIBRARY xrt_coreutil
    PATHS /opt/xilinx/xrt/lib
)

if(XRT_INCLUDE_DIR AND XRT_LIBRARY)
    set(XRT_FOUND TRUE)
    set(XRT_INCLUDE_DIRS ${XRT_INCLUDE_DIR})
    set(XRT_LIBRARIES ${XRT_LIBRARY})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XRT
    REQUIRED_VARS XRT_LIBRARY XRT_INCLUDE_DIR
)

mark_as_advanced(XRT_INCLUDE_DIR XRT_LIBRARY)
