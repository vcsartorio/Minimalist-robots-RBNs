#include "ci_kilobot_loop_functions.h"

#include <argos3/core/simulator/simulator.h>
#include <argos3/core/utility/configuration/argos_configuration.h>
#include <argos3/plugins/robots/kilobot/simulator/kilobot_entity.h>
#include <argos3/plugins/robots/kilobot/simulator/kilobot_measures.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <signal.h>
#include <cerrno>
#include <fcntl.h>
#include <cstdio>
#include <unistd.h>

#include <set>
#include <iomanip>

const std::string CONFIGURATION_KILOBOT_RW_ARENA_RADIUS = "arena_radius";
const std::string CONFIGURATION_KILOBOT_RW_NUM_ROBOTS = "num_robots";
const std::string CONFIGURATION_KILOBOT_RW_SAMPLING_PERIOD = "sampling_period_in_ticks";
const std::string CONFIGURATION_KILOBOT_RW_STRATEGY = "strategy";
const std::string CONFIGURATION_KILOBOT_RW_NODES = "num_nodes";
const std::string CONFIGURATION_KILOBOT_RW_COMM_RANGE = "communication_range";
const std::string CONFIGURATION_KILOBOT_RW_SPEED = "speed";
const std::string CONFIGURATION_KILOBOT_RW_ID_SIMULATION = "id_simulation";
const std::string CONFIGURATION_KILOBOT_RW_TARGET_PX = "target_posx";
const std::string CONFIGURATION_KILOBOT_RW_TARGET_PY = "target_posy";
const std::string CONFIGURATION_KILOBOT_RW_SAVE_POS = "save_position";


// const std::string CONFIGURATION_KILOBOT_RW_TARGET_RADIUS = "target_radius";

/****************************************/
/****************************************/

CIKilobotLoopFunctions::CIKilobotLoopFunctions() : m_pcFloor(NULL),
                                                   m_pcRNG(NULL),
                                                   m_unNumRobots(0),
                                                   m_tResults(),
                                                   m_fArenaRadius(1),
                                                   m_samplingPeriod(1),
                                                   m_fractionDiscovery(0),
                                                   m_fractionInformation(0),
                                                   m_internal_counter(0),
                                                   m_strategy("RBN"),
                                                   m_numNodes(0),
                                                   m_targetpx(0),
                                                   m_targetpy(0),
                                                   m_idSimu("001555"),
                                                   m_save_pos(0),
                                                   m_argos_tick_per_seconds(0),
                                                   m_argos_max_time(0),
                                                   m_random_seed(0),
                                                   m_target_position(0., 0., 0.),
                                                   m_communication_range(0.10f),
                                                   m_speed(1.0f)
                                                   
{
}

/****************************************/
/****************************************/

void CIKilobotLoopFunctions::Init(TConfigurationNode &t_node)
{
      m_pcFloor = &GetSpace().GetFloorEntity();

      m_pcRNG = CRandom::CreateRNG("argos");

      CSimulator &simulator = GetSimulator();
      TConfigurationNode &root = simulator.GetConfigurationRoot();
      TConfigurationNode &framework = GetNode(root, "framework");
      TConfigurationNode &experiment = GetNode(framework, "experiment");

      m_random_seed = simulator.GetRandomSeed();
      GetNodeAttributeOrDefault(experiment, "ticks_per_second", m_argos_tick_per_seconds, m_argos_tick_per_seconds);
      GetNodeAttributeOrDefault(experiment, "length", m_argos_max_time, m_argos_max_time);

      // /* Read parameters from configuration file */
      GetNodeAttributeOrDefault(t_node, CONFIGURATION_KILOBOT_RW_ARENA_RADIUS, m_fArenaRadius, m_fArenaRadius);
      GetNodeAttributeOrDefault(t_node, CONFIGURATION_KILOBOT_RW_NUM_ROBOTS, m_unNumRobots, m_unNumRobots);
      GetNodeAttributeOrDefault(t_node, CONFIGURATION_KILOBOT_RW_SAMPLING_PERIOD, m_samplingPeriod, m_samplingPeriod);
      GetNodeAttributeOrDefault(t_node, CONFIGURATION_KILOBOT_RW_STRATEGY, m_strategy, m_strategy);;
      GetNodeAttributeOrDefault(t_node, CONFIGURATION_KILOBOT_RW_NODES, m_numNodes, m_numNodes);
      GetNodeAttributeOrDefault(t_node, CONFIGURATION_KILOBOT_RW_COMM_RANGE, m_communication_range, m_communication_range);
      GetNodeAttributeOrDefault(t_node, CONFIGURATION_KILOBOT_RW_SPEED, m_speed, m_speed);
      GetNodeAttributeOrDefault(t_node, CONFIGURATION_KILOBOT_RW_ID_SIMULATION, m_idSimu, m_idSimu);
      GetNodeAttributeOrDefault(t_node, CONFIGURATION_KILOBOT_RW_TARGET_PX, m_targetpx, m_targetpx);
      GetNodeAttributeOrDefault(t_node, CONFIGURATION_KILOBOT_RW_TARGET_PY, m_targetpy, m_targetpy);
      GetNodeAttributeOrDefault(t_node, CONFIGURATION_KILOBOT_RW_SAVE_POS, m_save_pos, m_save_pos);

      // GetNodeAttributeOrDefault(t_node, CONFIGURATION_KILOBOT_RW_TARGET_RADIUS, m_fTargetRadius, m_fTargetRadius);

      ////////////////////////////////////////////////////////////////////////////////
      // CREATION AND POSITIONING OF THE ARENA WALLS (CIRCULAR ARENA)
      ////////////////////////////////////////////////////////////////////////////////
      Real wall_width = 0.01;
      Real wall_height = 0.05;

      std::ostringstream entity_id;
      Real m_unNumArenaWalls = 50; // TODO: this can be mande a configurable parameter
      CRadians wall_angle = CRadians::TWO_PI / m_unNumArenaWalls;
      CVector3 wall_size(wall_width, 2.0 * m_fArenaRadius * Tan(CRadians::PI / m_unNumArenaWalls), wall_height);
      for (UInt32 i = 0; i < m_unNumArenaWalls; i++)
      {
            entity_id.str("");
            entity_id << "wall_" << i;

            CRadians wall_rotation = wall_angle * i;
            CVector3 wall_position(m_fArenaRadius * Cos(wall_rotation), m_fArenaRadius * Sin(wall_rotation), 0);
            CQuaternion wall_orientation;
            wall_orientation.FromEulerAngles(wall_rotation, CRadians::ZERO, CRadians::ZERO);

            CBoxEntity *wall = new CBoxEntity(entity_id.str(), wall_position, wall_orientation, false, wall_size);
            AddEntity(*wall);
      }

      ////////////////////////////////////////////////////////////////////////////////
      // CREATION OF THE ROBOTS
      ////////////////////////////////////////////////////////////////////////////////
      std::ostringstream kilobot_id;
      CVector3 kilobot_position = CVector3(m_fArenaRadius, m_fArenaRadius, 0); // init position must be out of the arena
      kilobot_id.str("");
      kilobot_id << "0_target";
      CKilobotEntity *kilobot = new CKilobotEntity(kilobot_id.str(), "kbc_target", kilobot_position, CQuaternion(), m_communication_range);
      AddEntity(*kilobot);

      for (UInt32 i = 0; i < m_unNumRobots; i++)
      {
            kilobot_id.str("");
            kilobot_id << i + 1;
            CKilobotEntity *kilobot = new CKilobotEntity(kilobot_id.str(), "kbc_agent", kilobot_position, CQuaternion(), m_communication_range);
            AddEntity(*kilobot);
      }

      // kilobot->set_movable(false);

      m_cKilobots = GetSpace().GetEntitiesByType("kilobot");

      // Check RBN parameters
      checkParameters();

      // // set the experiment
      SetExperiment();
}

void CIKilobotLoopFunctions::DeleteSimConfig()
{
      std::string argos_path = "simulation_config/" + m_idSimu + "simulation_parameters.argos";
      remove(argos_path.c_str());
      LOG << "Deleting " + m_idSimu + "simulation_parameters.argos file\n";
}

void CIKilobotLoopFunctions::checkParameters()
{
      LOG << "Simulation " << m_idSimu << "\n";
      std::ifstream matrix("src/behaviors/parameters_rbn.txt");
      if (matrix.is_open()) {
            if (matrix.peek() == EOF) {
                  LOG << "Parameters file is empty!\n";
                  DeleteSimConfig();
                  LOG.Flush();
                  exit(11);
            }
            else {
                  std::string line;
                  LOG << "RBN parameters: ";
                  getline(matrix,line);
                  LOG << line << std::endl;
                  int num_nodes = std::stoi(line);
                  if (num_nodes != m_numNodes) {
                        LOG << "Num nodes doesnt match!\n";
                        DeleteSimConfig();
                        LOG.Flush();
                        exit(13);
                  }
                  LOG << "RBN connections: ";
                  for(int i = 0; i < num_nodes; i++){
                        getline(matrix,line);
                        LOG << line << " ";
                  }
                  LOG << std::endl;
                  LOG << "RBN boolean functions: ";
                  for(int i = 0; i < num_nodes; i++){
                        getline(matrix,line);
                        LOG << line << " ";
                  } 
                  LOG << std::endl;
            }
      }
      else {
            LOG << "Error when opening parameters file!\n";
            DeleteSimConfig();
            LOG.Flush();
            exit(12);
      }

      LOG.Flush();
}


/****************************************/
/****************************************/

void CIKilobotLoopFunctions::Reset()
{
      // LOG << "Reset!" << std::endl;
      SetExperiment();
}

/****************************************/
/****************************************/

// void CIKilobotLoopFunctions::Destroy()
// {
// }

/****************************************/
/****************************************/

void CIKilobotLoopFunctions::SetExperiment()
{
      // // initialise/reset the internal variables
      // LOG << "Set Experiment!" << std::endl;
      m_tResults.Reset();
      uint robot_num = 0;
      // CQuaternion target_random_rotation;

      CVector3 kilobot_displacement(-KILOBOT_ECCENTRICITY, 0, 0);
      // CRadians target_random_rotation_angle(m_pcRNG->Uniform(CRange<Real>(-CRadians::PI.GetValue(), CRadians::PI.GetValue())));
      // target_random_rotation.FromEulerAngles(target_random_rotation_angle, CRadians::ZERO, CRadians::ZERO);
      // Real target_rho = m_pcRNG->Uniform(CRange<Real>(0, m_fArenaRadius));
      // Real target_theta = m_pcRNG->Uniform(CRange<Real>(-CRadians::PI.GetValue(), CRadians::PI.GetValue()));
      // CVector3 target_random_position(target_rho * cos(target_theta), target_rho * sin(target_theta), 0);
      // target_random_position += kilobot_displacement.RotateZ(target_random_rotation_angle);

      for (CSpace::TMapPerType::iterator it = m_cKilobots.begin(); it != m_cKilobots.end(); ++it, robot_num++)
      {
            /* Get handle to kilobot entity and controller */
            CKilobotEntity &cKilobot = *any_cast<CKilobotEntity *>(it->second);

            /* Get a random rotation within the circular arena */
            CQuaternion random_rotation;
            CRadians random_rotation_angle(m_pcRNG->Uniform(CRange<Real>(-CRadians::PI.GetValue(), CRadians::PI.GetValue())));
            random_rotation.FromEulerAngles(random_rotation_angle, CRadians::ZERO, CRadians::ZERO);

            /* Get a non-colliding random position within the circular arena */
            bool distant_enough = false;
            UInt32 m_unMaxInitTrials = 1000;
            UInt32 un_init_trials = 0;

            Real min_distance = 0.25f;

            while (!distant_enough && (++un_init_trials < m_unMaxInitTrials))
            {
                  Real rho = m_pcRNG->Uniform(CRange<Real>(0, m_fArenaRadius));
                  Real theta = m_pcRNG->Uniform(CRange<Real>(-CRadians::PI.GetValue(), CRadians::PI.GetValue()));
                  CVector3 random_position(rho * cos(theta), rho * sin(theta), 0);
                  if ((robot_num == 0))
                  {
                        random_position.Set(m_targetpx, m_targetpy, 0);
                        m_target_position = random_position;
                        LOG << "target position is: " << random_position.GetX() << " " << random_position.GetY() << "\n";
                        LOG.Flush();
                  }

                  distant_enough = MoveEntity(cKilobot.GetEmbodiedEntity(), random_position + kilobot_displacement.RotateZ(random_rotation_angle), random_rotation, false);
                  
                  if (distant_enough)
                  {
                        CVector2 this_kilobot_xy_position(cKilobot.GetEmbodiedEntity().GetOriginAnchor().Position.GetX(), cKilobot.GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
                        uint other_robot_num = 0;
                        Real closest_distance = 10000000.0f;
                        for (CSpace::TMapPerType::iterator it_other = m_cKilobots.begin(); it_other != m_cKilobots.end() && other_robot_num < robot_num; ++it_other, other_robot_num++)
                        {
                              CKilobotEntity &other_kilobot = *any_cast<CKilobotEntity *>(it_other->second);
                              CVector2 other_kilobot_xy_position(other_kilobot.GetEmbodiedEntity().GetOriginAnchor().Position.GetX(), other_kilobot.GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
                              Real distance = (other_kilobot_xy_position - this_kilobot_xy_position).Length();
                              if (distance < closest_distance)
                              {
                                    closest_distance = distance;
                              }
                        }
                        Real random_refusal = m_pcRNG->Uniform(CRange<Real>(0.08, min_distance));
                        if (closest_distance < random_refusal)
                        {
                              distant_enough = false;
                              // printf("closest_distance = %f for robot %d try %d\n", closest_distance, robot_num, un_init_trials);
                        }
                  }

                  if (un_init_trials + 1 > m_unMaxInitTrials)
                  {
                        LOGERR << "Failed to move entity " << cKilobot.GetId() << " for  " << m_unMaxInitTrials << " trials" << std::endl;
                        LOGERR.Flush();
                  }
            }
            ////////////////////////
      }

      Real wall_width = 0.01;
      Real wall_height = 0.05;
      Real blockRadius = 0.016;
      std::ostringstream entity_id;
      Real numBlockWall = 50; // TODO: this can be mande a configurable parameter
      CRadians wall_angle = CRadians::TWO_PI / numBlockWall;
      CVector3 wall_size(wall_width, 2.0 * blockRadius * Tan(CRadians::PI / numBlockWall), wall_height);
      for (UInt32 i = 0; i < numBlockWall; i++)
      {
            entity_id.str("");
            entity_id << "wall_" << i + 100;

            CRadians wall_rotation = wall_angle * i;
            CVector3 wall_position(blockRadius * Cos(wall_rotation) + m_target_position[0], blockRadius * Sin(wall_rotation) + m_target_position[1], 0);
            CQuaternion wall_orientation;
            wall_orientation.FromEulerAngles(wall_rotation, CRadians::ZERO, CRadians::ZERO);

            CBoxEntity *wall = new CBoxEntity(entity_id.str(), wall_position, wall_orientation, false, wall_size);
            AddEntity(*wall);
      }

      UInt32 un_robot_index = 0;
      for (CSpace::TMapPerType::iterator it = m_cKilobots.begin(); it != m_cKilobots.end(); ++it, ++un_robot_index)
      {
            /* Get handle to kilobot entity and controller */
            CKilobotEntity &c_kilobot = *any_cast<CKilobotEntity *>(it->second);
            // printf("%s\n", c_kilobot.GetId().c_str());
            CVector2 c_kilobot_xy_position(c_kilobot.GetEmbodiedEntity().GetOriginAnchor().Position.GetX(), c_kilobot.GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
            m_cKilobotOriginalPositions.push_back(c_kilobot_xy_position);
            std::vector<CVector2> positions_init;

            positions_init.push_back(c_kilobot_xy_position);
            m_cKilobotPositions.push_back(positions_init);

            std::vector<Real> displacement_init;
            displacement_init.push_back((Real)0.0);
            m_cKilobotDisplacements.push_back(displacement_init);

            std::vector<int> times_init;
            times_init.push_back(0);
            times_init.push_back(0);
            m_cKilobotDiscoveryInformationTime.push_back(times_init);
      }
}

/****************************************/
/****************************************/

void CIKilobotLoopFunctions::PostStep()
{
      //Delete RBN parameters
      if(GetSpace().GetSimulationClock() == 1){
            DeleteSimConfig();
      }

      UInt32 un_robot_index = 0;
      UInt32 num_robots_with_discovery = 0;
      UInt32 num_robots_with_info = 0;
      m_internal_counter += 1;
      for (CSpace::TMapPerType::iterator it = m_cKilobots.begin(); it != m_cKilobots.end(); ++it, ++un_robot_index)
      {
            /* Get handle to kilobot entity and controller */
            CKilobotEntity &c_kilobot = *any_cast<CKilobotEntity *>(it->second);

            if (m_internal_counter == m_samplingPeriod)
            {
                  CVector2 c_kilobot_xy_position(c_kilobot.GetEmbodiedEntity().GetOriginAnchor().Position.GetX(), c_kilobot.GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

                  CVector2 originPlace = m_cKilobotOriginalPositions.at(un_robot_index);
                  Real squareDisplacement = (c_kilobot_xy_position - originPlace).SquareLength();

                  m_cKilobotPositions[un_robot_index].push_back(c_kilobot_xy_position);
                  m_cKilobotDisplacements[un_robot_index].push_back(squareDisplacement);
            }

            CCI_KilobotController &c_controller = dynamic_cast<CCI_KilobotController &>(c_kilobot.GetControllableEntity().GetController());
            int sharedMemFD = c_controller.GetSharedMemFD();

            kilobot_state_t *robotState;
            /* Resize shared memory area to contain the robot state, filling it with zeros */
            ftruncate(sharedMemFD, sizeof(kilobot_state_t));
            /* Get pointer to shared memory area */
            robotState =
                (kilobot_state_t *)mmap(NULL,
                                        sizeof(kilobot_state_t),
                                        PROT_READ,
                                        MAP_SHARED,
                                        sharedMemFD,
                                        0);
            if (robotState == MAP_FAILED)
            {
                  close(sharedMemFD);
                  exit(1);
            }

            message_t message = robotState->tx_message;

            int first_passage_time = 0;
            int first_info_time = 0;

            memcpy((void *)&first_info_time, (void *)&(message.data[1]), sizeof(int));
            memcpy((void *)&first_passage_time, (void *)&(message.data[5]), sizeof(int));

            // LOG << "first_info_time :" << first_info_time << " first_passage_time : " << first_passage_time << std::endl;
            // LOG.Flush();

            if (message.data[0] == 1 && first_passage_time != 0)
            {
                  num_robots_with_discovery += 1;
                  m_cKilobotDiscoveryInformationTime[un_robot_index][0] = first_passage_time;
            }
            if (message.data[0] == 1 && first_info_time != 0)
            {
                  num_robots_with_info += 1;
                  m_cKilobotDiscoveryInformationTime[un_robot_index][1] = first_info_time;
            }

            munmap(robotState, sizeof(kilobot_state_t));

      }
      if (m_internal_counter == m_samplingPeriod)
      {
            m_internal_counter = 0;
      }

      // Check results
      m_tResults.m_fFractionWithDiscovery = ((Real)num_robots_with_discovery) / ((Real)m_unNumRobots);
      m_tResults.m_fFractionWithInformation = ((Real)num_robots_with_info) / ((Real)m_unNumRobots);

      if (m_tResults.m_fFractionWithDiscovery != m_fractionDiscovery)
      {
            m_fractionDiscovery = m_tResults.m_fFractionWithDiscovery;
            LOG << "Fraction discovery is " << m_tResults.m_fFractionWithDiscovery << std::endl;
      }

      if (m_tResults.m_fFractionWithInformation != m_fractionInformation)
      {
            m_fractionInformation = m_tResults.m_fFractionWithInformation;
            LOG << "Fraction information is " << m_tResults.m_fFractionWithInformation << std::endl;
      }

      if ((num_robots_with_discovery == m_unNumRobots) && (m_tResults.m_unFullDiscoveryTime == 0))
            m_tResults.m_unFullDiscoveryTime = GetSpace().GetSimulationClock();

      if ((num_robots_with_info == m_unNumRobots) && (m_tResults.m_unFullInformationTime == 0))
            m_tResults.m_unFullInformationTime = GetSpace().GetSimulationClock();

      LOG.Flush();
}

/****************************************/
/****************************************/

const std::string currentDateTime()
{
      time_t now = time(0);
      struct tm tstruct;
      char buf[80];
      tstruct = *localtime(&now);
      strftime(buf, sizeof(buf), "%Y%m%d-%X-%M", &tstruct);

      return buf;
}

const std::string currentDate()
{
      time_t now = time(0);
      struct tm tstruct;
      char buf[80];
      tstruct = *localtime(&now);
      strftime(buf, sizeof(buf), "%Y-%m-%d", &tstruct);

      return buf;
}

void CIKilobotLoopFunctions::PostExperiment()
{
      std::string dateTime = currentDateTime();
      std::string date = currentDate();
      char numRobotStr[10];
      sprintf(numRobotStr, "%d", m_unNumRobots);
      char numNodesStr[10];
      sprintf(numNodesStr, "%d", m_numNodes);

      int disc_time = 0;
      int inf_time = 0;
      float n_robots_found = 0;
      float frac_disc = m_tResults.m_fFractionWithDiscovery;
      float frac_inf = m_tResults.m_fFractionWithInformation;
      for (uint i = 1; i <= m_unNumRobots; i++)
      {
            if ((m_cKilobotDiscoveryInformationTime[i])[0] != 0)
            {
                  n_robots_found++;
                  disc_time += (m_cKilobotDiscoveryInformationTime[i])[0];   
            }
            inf_time += (m_cKilobotDiscoveryInformationTime[i])[1];   
      }
      disc_time = disc_time/n_robots_found;
      inf_time = inf_time/m_unNumRobots;
      if(n_robots_found < 1){
            disc_time = 0;
            inf_time = 0;
            frac_disc = 0;
            frac_inf = 0;
      }

      std::cout << disc_time << " " << inf_time << " " << frac_disc << " " << frac_inf << " ";
      for (uint i = 1; i <= m_unNumRobots; i++)
            std::cout << (m_cKilobotDiscoveryInformationTime[i])[0] << " ";

      if (m_save_pos == 1)
      {
            std::stringstream pos_folder_path;
            pos_folder_path << "Data/Kilobot_position_data/" << m_strategy << "_robots#" << numRobotStr << "_arena#" << m_fArenaRadius
                  << "_nodes#" << numNodesStr << "_id#" << m_idSimu.substr(0, 4) << "_samplep#" << m_samplingPeriod;
            std::string folder = pos_folder_path.str();
            if (opendir(folder.c_str()) == NULL)
            {
                  const int dir_err = mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                  if (-1 == dir_err)
                  {
                        printf("Error creating directory!\n");
                        exit(1);
                  }
            }

            std::string position_file = folder + "/" + m_idSimu.substr(4) + "_trial_robots_position.tsv";
            std::ofstream of_3(position_file, std::ios::out);
            of_3 << "Robot id";

            for (uint j = 0; j < m_cKilobotDisplacements[1].size(); j++)
            {
                  int t = j * m_samplingPeriod;
                  // of_2 << "\tt = " << t;
                  of_3 << "\tt = " << t;
            }
            // of_2 << std::endl;
            of_3 << std::endl;

            for (uint i = 1; i <= m_unNumRobots; i++)
            {
                  of_3 << i;

                  for (uint j = 0; j < m_cKilobotPositions[i].size(); j++)
                  {
                        of_3 << '\t' << std::setprecision(3) << (m_cKilobotPositions[i])[j];
                  }
                  of_3 << std::endl;
            }
      }

      LOG.Flush();
}

bool CIKilobotLoopFunctions::IsExperimentFinished()
{
      if (m_tResults.m_unFullDiscoveryTime != 0 && m_save_pos == 0)
      {
            LOG << "Experiment Finished" << std::endl;
            LOG.Flush();
            return (m_tResults.m_unFullDiscoveryTime != 0);
      }
      return false;
}

/****************************************/
/****************************************/

REGISTER_LOOP_FUNCTIONS(CIKilobotLoopFunctions, "ci_kilobot_loop_functions")
