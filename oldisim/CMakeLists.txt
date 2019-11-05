cmake_minimum_required(VERSION 3.12)

project(oldisimlib)

add_library(oldisimlib
            src/AutoSnapshot.h
            src/CerealMapAsJSObject.h
            src/ChildConnection.cc
            src/ChildConnectionImpl.h
            src/ConnectionUtil.cc
            src/ConnectionUtil.h
            src/DriverNode.cc
            src/FanoutManager.cc
            src/FanoutManagerImpl.h
            src/ForcedEvTimer.h
            src/InternalCallbacks.h
            src/LeafNodeServer.cc
            src/Log.cc
            src/NodeThread.cc
            src/NodeThreadImpl.h
            src/ParentConnection.cc
            src/ParentConnectionImpl.cc
            src/ParentConnectionImpl.h
            src/ParentNodeServer.cc
            src/QueryContext.cc
            src/ResponseContext.cc
            src/TestDriver.cc
            src/TestDriverImpl.h)

target_include_directories(oldisimlib PUBLIC include/)
target_link_libraries(oldisimlib PRIVATE boost_1_53 cereal)
target_compile_features(oldisimlib PUBLIC cxx_std_11)
