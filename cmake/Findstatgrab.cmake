
find_path(STATGRAB_INCLUDE_PATH statgrab.h
	~/usr/include
	~/usr/.local/include
	~/.local/include
	~/usr/local/include
	/usr/include
	/usr/local/include
	thirdparty
)

set(STATGRAB_INCLUDE_DIR /usr/include)
set(STATGRAB_INCLUDE_PATH /usr/include)
#set(MIMALLOC_LIBRARY_DEBUG /usr/local/lib/libmimalloc-debug.so)
set(STATGRAB_LIBRARY /usr/lib/x86_64-linux-gnu/libstatgrab.so)

if(STATGRAB_INCLUDE_PATH)
	set(STATGRAB_FOUND TRUE)
	set(STATGRAB_INCLUDE_PATHS ${STATGRAB_INCLUDE_PATH} CACHE STRING "The include paths needed to use statgrab")
endif()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(STATGRAB DEFAULT_MSG
    STATGRAB_INCLUDE_DIR STATGRAB_LIBRARY)

IF(STATGRAB_FOUND)
  SET(STATGRAB_LIBRARIES optimized ${STATGRAB_LIBRARY})
  SET(STATGRAB_INCLUDE_DIRS ${STATGRAB_INCLUDE_DIR})
ENDIF(STATGRAB_FOUND)

mark_as_advanced(
	STATGRAB_INCLUDE_PATHS
)

# Generate appropriate messages
if(STATGRAB_FOUND)
    if(NOT STATGRAB_FIND_QUIETLY)
    	#message("-- Found STATGRAB: ${STATGRAB_INCLUDE_PATH}")
    endif(NOT STATGRAB_FIND_QUIETLY)
else(STATGRAB_FOUND)
    if(STATGRAB_FIND_REQUIRED)
	message(FATAL_ERROR "-- Could NOT find TooN (missing: STATGRAB_INCLUDE_PATH)")
    endif(STATGRAB_FIND_REQUIRED)
endif(STATGRAB_FOUND)
