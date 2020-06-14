set( LIBRARY_NAME "Lecture_donnees" )

add_library( ${LIBRARY_NAME} )
target_sources( ${LIBRARY_NAME} 
    PRIVATE
        LectureDonnees.cc
)
target_include_directories( ${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR} )
if ( ${CMAKE_SYSTEM_NAME} MATCHES "Linux" )
    target_link_libraries( ${LIBRARY_NAME} 
        PUBLIC
            cppnabla
            stdc++fs
    )
elseif( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )
	target_link_libraries( ${LIBRARY_NAME} 
        PUBLIC
            cppnabla
    )
    else()
    message( FATAL_ERROR "Not configured to be built on Windows" )
endif()
