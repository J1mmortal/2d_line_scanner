#include <AccelStepper.h>

// ============================
// PIN DEFINITIONS
// ============================

#define STEP_PIN        2   // To stepper driver STEP input + RS422 Step input
#define DIR_PIN         5   // To stepper driver DIR input + RS422 Dir input
#define ENDSTOP_PIN     9   // Microswitch (NO, to GND when triggered)

// ============================
// MOTION PARAMETERS
// ============================
#define MAXIMUM_SPEED           1000 //max speed, keep at 1000 for reliability

#define CALIBRATION_SPEED_NOM   800   // Homing speed 
#define CALIBRATION_SPEED_SLOW  50   // Slower homing speed
#define STEPS_CAL               40   // Homing distance (second hit)

#define NORMAL_TEST_SPEED       1000    // Constant-speed scan
#define TOTAL_SCAN_STEPS        9000   // Steps for one scan sweep
#define ACC_NORMAL_TEST         5000 // acceleration for testing


int endStopState = 0;
char incomingCmd = 0;
AccelStepper Xaxis(1, STEP_PIN, DIR_PIN);
//AccelStepper Yaxis(1, )

void calibration() {
  //setting the initial parameters for calibration
  Serial.println("calibration started");
  Xaxis.setSpeed(CALIBRATION_SPEED_NOM);

  //calibration checking loop
  while (digitalRead(ENDSTOP_PIN) == HIGH) {
    Xaxis.runSpeed();
  }

  //first time reaching endstop, calling that zero, and moving slightly forward for second test
  Xaxis.setCurrentPosition(0);
  Serial.println("first calibration finished");
  delay(50);
  Xaxis.moveTo(-STEPS_CAL);
  Xaxis.setSpeed(-CALIBRATION_SPEED_SLOW);
    while (Xaxis.distanceToGo() != 0) {
    Xaxis.runSpeed();
  }

  
  //slower calibration afterwards
  Xaxis.setSpeed(CALIBRATION_SPEED_SLOW);
  while (digitalRead(ENDSTOP_PIN) == HIGH) {
    Xaxis.runSpeed();
  }
  Xaxis.setCurrentPosition(100);
  Xaxis.moveTo(0);
  Xaxis.setSpeed(-CALIBRATION_SPEED_SLOW);
  while (Xaxis.distanceToGo() != 0) {
    Xaxis.runSpeedToPosition();
  }
  Serial.println("full calibration finished");
}

void normaltest() {
  Xaxis.setMaxSpeed(NORMAL_TEST_SPEED);
  Xaxis.moveTo(-4000);
  while (Xaxis.distanceToGo() != 0) {
    Xaxis.run();
  }
  delay(1000);
  Xaxis.moveTo(-8500);
  while (Xaxis.distanceToGo() != 0) {
    Xaxis.run();
  }
  delay(3000);
  Xaxis.moveTo(-4000);
  while (Xaxis.distanceToGo() != 0) {
    Xaxis.run();
  }
  
}
void return_home() {
  Xaxis.setMaxSpeed(NORMAL_TEST_SPEED);
  Xaxis.moveTo(0);
  while (Xaxis.distanceToGo() != 0) {
    Xaxis.run();
  }
}

void moveForward() {
  Xaxis.setMaxSpeed(-NORMAL_TEST_SPEED);
  Xaxis.move(-100);
  while (Xaxis.distanceToGo() != 0) {
    Xaxis.run();
  }
  
}

void moveBackward() {
  Xaxis.setMaxSpeed(NORMAL_TEST_SPEED);
  Xaxis.move(100);
  while (Xaxis.distanceToGo() != 0) {
    Xaxis.run();
  }
  
}

void alt_test_1() {
  Xaxis.setMaxSpeed(NORMAL_TEST_SPEED);
  Xaxis.moveTo(-4000);
  while (Xaxis.distanceToGo() != 0) {
    Xaxis.run();
  }
  delay(1000);
  Xaxis.moveTo(-6500);
  while (Xaxis.distanceToGo() != 0) {
    Xaxis.runSpeed();
  }
  Xaxis.moveTo(-8500);
  Xaxis.setMaxSpeed(NORMAL_TEST_SPEED/2);
  while (Xaxis.distanceToGo() != 0) {
    Xaxis.runSpeed();
  }
  delay(3000);
  Xaxis.moveTo(-4000);
  Xaxis.setMaxSpeed(NORMAL_TEST_SPEED);
  while (Xaxis.distanceToGo() != 0) {
    Xaxis.run();
  }
  

}

void setup() {
  //setting the pinmodes
  Serial.begin(9600);

  Serial.println("setup started");
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(ENDSTOP_PIN, INPUT_PULLUP);
  Xaxis.setMaxSpeed(MAXIMUM_SPEED);
  Xaxis.setAcceleration(ACC_NORMAL_TEST);

  Serial.println("calibration()");
  calibration();
  Serial.println("c: calibration \nh: return home \nn: normal test \na: alternative test 1 \nb: backward \nf: forward");
}

void moveToPoisition() {
  Serial.println('n');
}

void checkSerialCommand() {
  if (Serial.available() > 0) {
    incomingCmd = Serial.read();

    if (incomingCmd == 'c') {
      Serial.println("Serial command: calibration()");
      calibration();
    } 
    else if (incomingCmd == 'h') {
      Serial.println("Serial command: return_home()");
      return_home();
    } 
    else if (incomingCmd == 'n') {
      Serial.println("Serial command: normaltest()");
      normaltest();
    }
        else if (incomingCmd == 'a') {
      Serial.println("Serial command: alt_test_1()");
      alt_test_1();
    }
    else if (incomingCmd == 'b') {
      Serial.println("Serial command: moveBackward()");
      moveBackward();
    }
    else if (incomingCmd == 'f') {
      Serial.println("Serial command: moveForward()");
      moveForward();
    } 
    else {
      Serial.println("Unknown command: ");
      Serial.println(incomingCmd);
      Serial.println("c: calibration \nh: return home \nn: normal test \na: alternative test 1 \nb: backward \nf: forward");

    }
  }
}


void loop() {
  checkSerialCommand();
  // put your main code here, to run repeatedly:
  
}
