cmake_minimum_required(VERSION 2.8.12)


add_custom_target(
  kilobot_real
  ALL
  COMMAND make all
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

