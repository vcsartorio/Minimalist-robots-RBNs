/*@authors vtrianni and cdimidov*/

#include "kilolib.h"
// ****************************************

#include "enum.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>

#if REAL
#define DEBUG
#include <stdarg.h>
#include "debug.h"
#include <avr/eeprom.h>
#else
#include <inttypes.h>
#endif

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif
#define M2_PI (2.0*M_PI)

#define MAX_TIME 100000000

/* Enum for different motion types */
typedef enum
{
  STOP = 0,
  FORWARD,
  TURN_LEFT,
  TURN_RIGHT,
} motion_t;

/*variable that memorizes the first passage time and the time*/
uint32_t f_p_t = 0; // first passage time
uint32_t f_i_t = 0; // first moment when i received the information

/*Max number of ticks*/
const double max_time = MAX_TIME;

/*STD*/
const double std_motion_steps = 5 * 16;

/* current motion type */
motion_t current_motion_type = STOP;

/*Message send to the others agents*/
message_t messageA;

/* Flag for decision to send a word */
bool sending_msg = false;

/*Flag for the existence of information*/
bool information_target_id = false;

/* Flag : send the id and first passage time to target */
bool on_target_flag = false;

/* counters for motion, turning and broadcasting */
uint32_t turning_ticks = 0;
const uint8_t max_turning_ticks = 160;
uint8_t max_straight_ticks = 0;
unsigned int straight_ticks = 0; /* 5*16; */ /* n_t*0.5*32 */
uint32_t last_motion_ticks = 0;
const uint8_t max_broadcast_ticks = 2 * 16; /* n_s*0.5*32 */
uint32_t last_broadcast_ticks = 0;
const uint16_t max_info_ticks = 7 * 16;
uint32_t last_info_ticks = 0;

/* Set Random Boolean Network parameters*/
int num_nodes;
int num_boolfunctions;
// int num_virtualsensors = 4;
// int found_robots = 0;
int max_bn_step_value;
char* parameters_file = "src/behaviors/parameters_rbn.txt";
char* copy_file = "src/behaviors/file2.txt";
int cont_function_call = 0;
char *cont_call_file = "cont_call_function.txt";

/* Creating Random Boolean Network struct */
struct RBN{
  int *nodes_state;
  int *connections_number;
	int **connections;
	int **booleanfunctions;
};
struct RBN individual_rbn;

/*-------------------------------------------------------------------*/
/* Random Boolean Network functions                                  */
/*-------------------------------------------------------------------*/
void allocateRBN(){
  individual_rbn.nodes_state = malloc(sizeof(int)*num_nodes);
  individual_rbn.connections_number = malloc(sizeof(int)*num_nodes);
  individual_rbn.connections = malloc(sizeof(int*)*num_nodes);
	for(int i = 0; i < num_nodes; i++)
		individual_rbn.connections[i] = malloc(sizeof(int)*num_nodes);
  individual_rbn.booleanfunctions = malloc(sizeof(int*)*num_nodes);
	for(int i = 0; i < num_nodes; i++)
		individual_rbn.booleanfunctions[i] = malloc(sizeof(int)*num_boolfunctions);
}

void calculateConnectionsNum(){
    // Calculate the number of connections each node has
	for(int i = 0; i < num_nodes; i++){
		individual_rbn.connections_number[i] = 0;
		for(int j = 0; j < num_nodes; j++){
			if(individual_rbn.connections[i][j] > 0)
				individual_rbn.connections_number[i]++;
		}
    }
}

void setRBNparameters(){
	// Open and read the parameters file
	char ch;
	FILE *fptr = fopen(parameters_file, "r");
	if(fptr == NULL){
		printf("kilobot_rbn - Couldnt open parameters_rbn file!\n");
		exit(0);
	}
  else{
    fseek(fptr, 0, SEEK_END); 
	  long fsize = ftell(fptr);
    if (fsize == 0){
      printf("kilobot_rbn - File is empty!\n");
      exit(0);
    }
  }
  // Set number of nodes in rbn
  fseek(fptr, 0, SEEK_SET);
  num_nodes = (int)fgetc(fptr) - '0';
  ch = fgetc(fptr);
  if (ch != '\n'){
    num_nodes = num_nodes*10 + ((int)ch - '0');
    ch = fgetc(fptr);
  }
  num_boolfunctions = num_nodes - 1;
  allocateRBN();
  max_bn_step_value = pow(2, num_nodes/2);

  // Set the nodes connection and booleanfuntions value
  for(int i = 0; i < num_nodes; i++){
    for(int j = 0; j < num_nodes; j++){
      ch = fgetc(fptr);
      if(ch != '\n')
        individual_rbn.connections[i][j] = (int)ch - '0';
      else
        j = -1;
    }
  }
  // Set the nodes boolean functions
  for(int i = 0; i < num_nodes; i++){
    for(int j = 0; j < num_boolfunctions; j++){
      ch = fgetc(fptr);
      if(ch != '\n')
        individual_rbn.booleanfunctions[i][j] = (int)ch - '0';
      else
        j = -1;
    }
  }
  // Jump to the robot nodes initial state line
  int robot_id = atoi(kilo_str_id);
  for(int i = 0; i < robot_id - 1; i++){ 
    for(int j = 0; j < num_nodes; j++){
      ch = fgetc(fptr);
      if(ch == '\n')
        j = -1;
    }
  }
  // Set the nodes initial state
  for(int i = 0; i < num_nodes; i++){
    ch = fgetc(fptr);
    if (ch != '\n')
		  individual_rbn.nodes_state[i] = (int)ch - '0';
    else
      i = -1;
  }
  // printf("Robot %d initiate with: ", robot_id);
  // for(int i = 0; i < num_nodes; i++)
  //   printf("%d", individual_rbn.nodes_state[i]);
  // printf("\n");

  fclose(fptr);
  calculateConnectionsNum();
}

void freeingRBN(){
	for(int i = 0; i < num_nodes; i++)
		free(individual_rbn.connections[i]);
	for(int i = 0; i < num_boolfunctions; i++)
		free(individual_rbn.booleanfunctions[i]);
	free(individual_rbn.nodes_state);
  free(individual_rbn.connections_number);
	free(individual_rbn.booleanfunctions);
  free(individual_rbn.connections);
}

int logicmapRBN(int boolfunction, int node_a, int node_b){
	switch(boolfunction){
		case 0:
			return node_a & node_b; /*AND*/
		case 1:
			return node_a | node_b; /*OR*/
		case 2:
			return node_a ^ node_b; /*XOR*/
		case 3:
			return !(node_a ^ node_b); /*XNOR*/
		case 4:
			return !(node_a & node_b); /*NAND*/
		case 5:
			return !(node_a | node_b); /*NOR*/
		default:
			printf("kilobot_rbn - Error in logicmap = %d\n", boolfunction);
			break;
	}
	return 0;
}

// void updateVirtualSensors(){
//   int begin_vsensor = (num_nodes - num_virtualsensors);
// 	if(found_robots > 4)
//     found_robots = 4;
//   for(int i = begin_vsensor; i < num_nodes; i++){
//     if (i < (begin_vsensor + found_robots))
//       individual_rbn.nodes_state[i] = 1;
//     else
//       individual_rbn.nodes_state[i] = 0;
//   }
//   found_robots = 0;
// }

void printNodeValues(){
  if (atoi(kilo_str_id) == 10){
    for(int i = 0; i < num_nodes; i++){
      printf("%d", individual_rbn.nodes_state[i]);
    }
    printf("\n");
  }
}

void printAngleAndStepLenghtValues(double angle, int step_lenght, int direction){
  if (direction == 0)
    printf("Angle: %f - Direction: Left - Step lenght: %d\n", angle, step_lenght);
  else
    printf("Angle: %f - Direction: Right - Step lenght: %d\n", angle, step_lenght);
}

void writeAngleAndStepLenghtValues(double angle, int step_lenght, int direction){
    char *filename = "src/behaviors/resultsAngleSL.txt";

    FILE *fp = fopen(filename, "a");
    if (fp == NULL)
    {
        printf("Error opening the file %s", filename);
        return;
    }
    if (direction == 0)
        fprintf(fp, "Angle: %f - Direction: Left - Step lenght: %d\n", angle, step_lenght);
    else
        fprintf(fp, "Angle: %f - Direction: Right - Step lenght: %d\n", angle, step_lenght);

    fclose(fp);
    return;
}

void calculateRbnNextStep(){
  // Update virtual sensor
  //updateVirtualSensors();
  // Calculate the next step for each node
  int* new_node_states = malloc(sizeof(int)*num_nodes);
  for(int i = 0; i < num_nodes; i++){
    // Get the state of each node connected to itself
    if(individual_rbn.connections_number[i] > 1){
      int* connected_nodes_state = malloc(sizeof(int)*num_nodes);
      int cont_nodes = 0;
      for(int j = 0; j < num_nodes; j++){
        if(individual_rbn.connections[i][j] == 1){
          connected_nodes_state[cont_nodes] = individual_rbn.nodes_state[j];
          cont_nodes++;
        }
        else if(individual_rbn.connections[i][j] == 2){
          if(individual_rbn.nodes_state[j] == 0)
            connected_nodes_state[cont_nodes] = 1;
          else 
            connected_nodes_state[cont_nodes] = 0;
          cont_nodes++;
        }
      }
      // Calculate next node state
      int flogic_num = individual_rbn.connections_number[i] - 1;
      int connected_state_len = individual_rbn.connections_number[i] + flogic_num;
      int *connected_state = malloc(sizeof(int)*(connected_state_len));
      for(int j = 0; j < individual_rbn.connections_number[i]; j++)
        connected_state[j] = connected_nodes_state[j];
      int cont_states = 0;
      for(int j = 0; j < flogic_num; j++){
        int aux_flogic = individual_rbn.booleanfunctions[i][j];
        int new_state = logicmapRBN(aux_flogic, connected_state[cont_states], connected_state[cont_states+1]);
        connected_state[individual_rbn.connections_number[i] + j] = new_state;
        cont_states += 2;
      }
      new_node_states[i] = connected_state[connected_state_len - 1];
      free(connected_nodes_state);
      free(connected_state);
    }
    else
      new_node_states[i] = individual_rbn.nodes_state[i];
  }
  // Set the new states for each node
  for(int i = 0; i < num_nodes; i++)
    individual_rbn.nodes_state[i] = new_node_states[i];
  free(new_node_states);
  // printNodeValues();
}

void getValues(double *angle, int *step_lenght, int *direction){
  // Get the angle
  int angle_direction = 0;
  int num_values = num_nodes/2;
  for(int i = 0; i < num_values; i++){
    if(individual_rbn.nodes_state[i] == 1)
      angle_direction += pow(2, i);
  }
  *angle = angle_direction - ((pow(2, num_values-1) - 1));
  *angle = (*angle / (float)(pow(2, num_values-1) - 1));
  if (*angle < 0){
    *direction = 0;
    *angle = abs(*angle);
  }
  else{
    *direction = 1;
  }

  // Get the step lenght (distance it goes forward)
  *step_lenght = 0;
  for(int i = num_values; i < num_nodes; i++){
    if(individual_rbn.nodes_state[i] == 1)
      *step_lenght += pow(2, i - num_values);
  }
  int max_encode_value = pow(2,10);
  float step_percentage = (*step_lenght/(float)max_bn_step_value);
  *step_lenght = step_percentage * max_encode_value;

  // if (atoi(kilo_str_id) == 10){
  //   // printAngleAndStepLenghtValues(*angle, *step_lenght, *direction);
  //   writeAngleAndStepLenghtValues(*angle, *step_lenght, *direction);
  // }
}

/*-------------------------------------------------------------------*/
/* Printf function                                                   */
/*-------------------------------------------------------------------*/

void my_printf(const char *fmt, ...)
{
#if REAL
  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);

  va_end(args);
#endif
}

/*-------------------------------------------------------------------*/
/* Function for setting the motor speed                              */
/*-------------------------------------------------------------------*/
void set_motion(motion_t new_motion_type)
{
  if (kilo_ticks < max_time)
  {
    if (current_motion_type != new_motion_type)
    {
      switch (new_motion_type)
      {
      case FORWARD:
        spinup_motors();
        set_motors(kilo_straight_left, kilo_straight_right);
        break;
      case TURN_LEFT:
        spinup_motors();
        set_motors(kilo_turn_left, 0);
        break;
      case TURN_RIGHT:
        spinup_motors();
        set_motors(0, kilo_turn_right);
        break;
      case STOP:
      default:
        set_motors(0, 0);
      }
      current_motion_type = new_motion_type;
    }
  }
  else
  {
    set_motors(0, 0);
  }
}

/*-------------------------------------------------------------------*/
/* Init function                                                     */
/*-------------------------------------------------------------------*/
void setup()
{

  /* Initialise LED and motors */
  set_motors(0, 0);
  /* Initialise random seed */
  uint8_t seed = rand_hard();
  rand_seed(seed);
  srand(seed);
  f_i_t = 0;
  f_p_t = 0;
  /*Compute the message CRC value for the Target_id*/
  messageA.data[0] = (uint8_t)id_robot;
  /* Compute the message CRC value for the id */
  messageA.data[1] = 0; // 0 I don't have the message
  messageA.type = NORMAL;
  memset(&(messageA.data[1]), 0, 8);
  messageA.crc = message_crc(&messageA);

  /* Initialise motion variables */
  set_motion(FORWARD);
}
/*-------------------------------------------------------------------*/
/* Callback function for message transmission                        */
/*-------------------------------------------------------------------*/
message_t *message_tx()
{
  if (sending_msg)
  {
    return &messageA;
  }
  return 0;
}

/*-------------------------------------------------------------------*/
/* Callback function for successful transmission                     */
/*-------------------------------------------------------------------*/
void tx_message_success()
{
  sending_msg = false;
}

/*-------------------------------------------------------------------*/
/* Callback function for message reception                           */
/*-------------------------------------------------------------------*/
void message_rx(message_t *msg, distance_measurement_t *d)
{
  uint8_t cur_distance = estimate_distance(d);
  if (cur_distance > 100)
  {
    return;
  }
  uint8_t agent_type = msg->data[0] & 0x01;
  // my_printf("%u"
  // "\n",
  // agent_type);
  //if the message received is 0 so the id of the target set color red
  if (agent_type == id_target)
  {
    if (f_p_t == 0)
    {
      f_p_t = kilo_ticks;
      memcpy(((void *)(&(messageA.data[5]))), &f_p_t, sizeof(int32_t));
      messageA.crc = message_crc(&messageA);

      // my_printf("The kilobot is on target for the first time\n");
      // my_printf("%" PRIu32 "\n", f_p_t);
      information_target_id = true;
      set_color(RGB(3, 0, 3));
    }
    if (f_i_t == 0)
    {
      f_i_t = kilo_ticks;
      memcpy(((void *)(&(messageA.data[1]))), &f_i_t, sizeof(int32_t));
      messageA.crc = message_crc(&messageA);

      // my_printf("The kilobot has the information but cross the target for the first time \n");
      // my_printf("%" PRIu32 "\n", f_i_t);
    }
  }
  //if the message received is 1 so the id of the target set color green
  else if (agent_type == id_robot)
  {
    // if ((kilo_ticks - last_motion_ticks) < 10)
    //   found_robots++;
    if (f_i_t == 0)
    {
      f_i_t = kilo_ticks;
      memcpy(((void *)(&(messageA.data[1]))), &f_i_t, sizeof(int32_t));
      messageA.crc = message_crc(&messageA);
      // my_printf("The kilobot receive information from the other robot\n");
      // my_printf("%" PRIu32 "\n", f_i_t);
      information_target_id = true;
      set_color(RGB(0, 3, 0));
    }
  }
}

/*-------------------------------------------------------------------*/
/* Function to send a message                                        */
/*-------------------------------------------------------------------*/
void broadcast()
{

  if (information_target_id && !sending_msg && kilo_ticks > last_broadcast_ticks + max_broadcast_ticks)
  {
    last_broadcast_ticks = kilo_ticks;
    sending_msg = true;
  }
}
/*----------------------------------------------------------------------*/
/* Function implementing the correlated random walk and levy random walk*/
/*----------------------------------------------------------------------*/
void random_walk(){
  if(kilo_ticks < max_time)
  {
    switch (current_motion_type)
    {
      case TURN_LEFT:
      case TURN_RIGHT:
        if(kilo_ticks > last_motion_ticks + turning_ticks)
        {
          /* start moving forward */
          last_motion_ticks = kilo_ticks;
          set_motion(FORWARD);
        }
        break;
      case FORWARD:
        if(kilo_ticks > last_motion_ticks + straight_ticks)
        {
          /* perform a random turn */
          last_motion_ticks = kilo_ticks;
          double angle;
          int step_lenght;
          int direction;
          getValues(&angle, &step_lenght, &direction);
          if (direction == 0)
            set_motion(TURN_LEFT);
          else
            set_motion(TURN_RIGHT);
          turning_ticks = (uint32_t)(angle * max_turning_ticks);
          straight_ticks = (uint32_t)(step_lenght);
          // my_printf("%u" "\n", straight_ticks);
          calculateRbnNextStep();
        }
        break;
      case STOP:
      default:
        set_motion(STOP);
    }
  }
  else
  {
    set_motion(STOP);
  }
}

void kilobotinfo()
{
  if (kilo_ticks > last_info_ticks + max_info_ticks)
  {
    last_info_ticks = kilo_ticks;
    my_printf("FPT:");
    my_printf("%" PRIu32 "\n", f_p_t);
    my_printf("Interaction Time:");
    my_printf("%" PRIu32 "\n", f_i_t);
  }
}

void check_reset()
{
  if (kilo_ticks == 0) // NOT THE RIGHT TEST, kilo_tick doesnt reinitiate?
  {
    setup();
  }
}

/*-------------------------------------------------------------------*/
/* Main loop                                                         */
/*-------------------------------------------------------------------*/
void loop()
{
  check_reset();
  random_walk();
  broadcast();
  kilobotinfo();
}

/*-------------------------------------------------------------------*/
/* Main function                                                     */
/*-------------------------------------------------------------------*/
int main()
{
  kilo_init();
  kilo_message_rx = message_rx;
  kilo_message_tx = message_tx;
  kilo_message_tx_success = tx_message_success;

#if REAL
  debug_init();
#endif

  /* start main loop */
  srand(1);
  setRBNparameters();
  kilo_start(setup, loop);
  freeingRBN();

  return 0;
}
