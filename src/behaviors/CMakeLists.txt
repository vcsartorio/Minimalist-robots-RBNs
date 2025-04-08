# THIS IS ONLY FOR SIMULATION -> TODO deport it to a behaviors_simulation folder

Set(REAL 0)
if(ARGOS_BUILD_FOR_SIMULATOR)
include_directories(/usr/local/include/argos3/plugins/robots/kilobot/control_interface /usr/local/include/argos3/plugins/robots/kilobot/simulator)


  add_executable(kilobotRBN kilobot_rbn.c)
  target_link_libraries(kilobotRBN argos3plugin_simulator_kilolib rt m)

  add_executable(kilobotRBNencoded kilobot_rbn_encoded.c)
  target_link_libraries(kilobotRBNencoded argos3plugin_simulator_kilolib rt m)

  add_executable(kilobotRBNnewEncode kilobot_rbn_new_encode.c)
  target_link_libraries(kilobotRBNnewEncode argos3plugin_simulator_kilolib rt m)


  add_executable(Target Target.c)
  target_link_libraries(Target argos3plugin_simulator_kilolib rt m)



endif(ARGOS_BUILD_FOR_SIMULATOR)


