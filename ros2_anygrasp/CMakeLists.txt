cmake_minimum_required(VERSION 3.5)
project(ros2_anygrasp)

# Default to C++14 (not really used here, but standard pattern)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Find build tools / dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate interfaces (srv)
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetAnyGraspPoses.srv"
  DEPENDENCIES geometry_msgs
)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install your Python nodes as executables
install(
  PROGRAMS
    ros2_anygrasp/grasp_planner.py
    ros2_anygrasp/anygrasp_grasp_service_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Export dependencies
ament_export_dependencies(
  rclpy
  sensor_msgs
  geometry_msgs
  cv_bridge
  message_filters
  rosidl_default_runtime
)

ament_package()
