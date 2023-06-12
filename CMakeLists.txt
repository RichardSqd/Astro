cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

PROJECT(Raycasting)
SET(VTK_DIR D:/VTK-9.1.0/bin)
SET(CMAKE_VERBOSE_MAKEFILE ON)
find_package(VTK REQUIRED)

add_executable(Raycasting pathtracer.cxx)


target_link_libraries(Raycasting glu32)
target_link_libraries(Raycasting opengl32)
if(VTK_LIBRARIES)
target_link_libraries(Raycasting ${VTK_LIBRARIES})
else()
target_link_libraries(Raycasting vtkHybrid)
endif()

vtk_module_autoinit(
  TARGETS Raycasting
  MODULES ${VTK_LIBRARIES})