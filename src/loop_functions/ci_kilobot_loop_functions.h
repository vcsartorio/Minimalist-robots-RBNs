#ifndef CI_KILOBOT_LOOP_FUNCTIONS_H
#define CI_KILOBOT_LOOP_FUNCTIONS_H

#include <argos3/core/simulator/loop_functions.h>
#include <argos3/core/simulator/entity/floor_entity.h>
#include <argos3/core/utility/math/range.h>
#include <argos3/core/utility/math/rng.h>
#include <argos3/core/control_interface/ci_controller.h>
#include <argos3/plugins/robots/generic/control_interface/ci_differential_steering_actuator.h>
#include <argos3/plugins/robots/generic/control_interface/ci_leds_actuator.h>
#include <argos3/plugins/robots/kilobot/control_interface/ci_kilobot_communication_actuator.h>
#include <argos3/plugins/robots/kilobot/control_interface/ci_kilobot_communication_sensor.h>
#include <argos3/plugins/robots/kilobot/control_interface/ci_kilobot_controller.h>

#include <argos3/core/utility/configuration/argos_configuration.h>
#include <argos3/core/utility/math/vector2.h>
#include <algorithm>
#include <argos3/core/utility/math/rng.h>
#include <argos3/core/utility/logging/argos_log.h>
#include <vector>
#include <argos3/plugins/simulator/entities/box_entity.h>
#include <sys/types.h>
#include <dirent.h>

using namespace argos;

////////////////////////////////////////////////////////////////
// Results struct
////////////////////////////////////////////////////////////////
struct TRWResults
{
  UInt32 m_unFullDiscoveryTime;
  UInt32 m_unFullInformationTime;
  Real m_fFractionWithInformation;
  Real m_fFractionWithDiscovery;

  TRWResults()
  {
    m_unFullDiscoveryTime = 0;
    m_unFullInformationTime = 0;
    m_fFractionWithInformation = 0.0;
    m_fFractionWithDiscovery = 0.0;
  }

  void Reset()
  {
    m_unFullDiscoveryTime = 0;
    m_unFullInformationTime = 0;
    m_fFractionWithInformation = 0.0;
    m_fFractionWithDiscovery = 0.0;
  }

  friend std::ostream &operator<<(std::ostream &os, const TRWResults &t_results)
  {
    os << t_results.m_unFullDiscoveryTime << " "
       << t_results.m_unFullInformationTime << " "
       << t_results.m_fFractionWithInformation << " "
       << t_results.m_fFractionWithDiscovery;
    return os;
  }
};

////////////////////////////////////////////////////////////////
// RW Loop Functions struct
////////////////////////////////////////////////////////////////

class CIKilobotLoopFunctions : public CLoopFunctions
{
public:
  CIKilobotLoopFunctions();
  virtual ~CIKilobotLoopFunctions() {}

  virtual void Init(TConfigurationNode &t_tree);
  virtual void Reset();
  //  virtual void Destroy();
  //  virtual CColor GetFloorColor(const CVector2& c_position_on_plane);
  virtual void PostStep();
  virtual void DeleteSimConfig();
  virtual void checkParameters();

  virtual bool IsExperimentFinished();
  virtual void SetExperiment();
  virtual void PostExperiment();

  const UInt32 GetNumRobots() const
  {
    return m_unNumRobots;
  };
  void SetNumRobots(const UInt32 un_num_robots) { m_unNumRobots = un_num_robots; };

  inline const TRWResults &GetResults() const { return m_tResults; };

private:
  CFloorEntity *m_pcFloor;
  CRandom::CRNG *m_pcRNG;
  UInt32 m_unNumRobots;
  TRWResults m_tResults;

  Real m_fArenaRadius;
  //  CVector2 m_cTargetPosition;
  //  Real m_fTargetRadius;
  int m_samplingPeriod;

  Real m_fractionDiscovery;
  Real m_fractionInformation;
  int m_internal_counter;
  std::string m_strategy;
  int m_numNodes;
  Real m_targetpx;
  Real m_targetpy;
  std::string m_idSimu;
  int m_save_pos;

  Real m_argos_tick_per_seconds;
  Real m_argos_max_time;

  uint m_random_seed;

  CVector3 m_target_position;
  Real m_communication_range;
  Real m_speed;

  CSpace::TMapPerType m_cKilobots;
  std::vector<CVector2> m_cKilobotOriginalPositions;
  std::vector<std::vector<Real>> m_cKilobotDisplacements;
  std::vector<std::vector<CVector2>> m_cKilobotPositions;
  std::vector<std::vector<int>> m_cKilobotDiscoveryInformationTime;
};

#endif
