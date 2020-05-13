#include "mbed.h"
#include "uLCD_4DGL.h"
#include "accelerometer_handler.h"

#include <cmath>
#include "DA7212.h"

#include "config.h"

#include "magic_wand_model_data.h"


#include "tensorflow/lite/c/common.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "tensorflow/lite/micro/micro_interpreter.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/version.h"

#define bufferLength (60)
#define signalLength (60)

DA7212 audio;
int song[signalLength];
int noteLength[bufferLength];
int stop;

int16_t waveform[kAudioTxBufferSize];

char serialInBuffer[bufferLength];

int serialCount = 0;
//int16_t waveform[kAudioTxBufferSize];

// int song[42];

//DigitalOut green_led(LED2);
Serial pc( USBTX, USBRX );
uLCD_4DGL uLCD(D1, D0, D2);
int Song = 0;
int Mode = 0;
Thread t_DisplaySongInfo;
Thread t_ModeSelection;
// Thread t_DNN(osPriorityNormal, 120 * 1024);
Thread t_DNN;
Thread t_Confirm;
Thread t_ChangeSong;
InterruptIn sw2(SW2);
InterruptIn sw3(SW3);
Thread t;
Thread t_CallAudio;
EventQueue q_DisplaySongInfo;
EventQueue q_ModeSelection;
EventQueue q_DNN;
EventQueue q_Confirm;
EventQueue q_ChangeSong;
EventQueue queue;
EventQueue q_CallAudio;
// The gesture index of the prediction

int gesture_index;
int gesture_index_pre;
int Mode_pre;
int ModeOrSong = 1;
int Song_pre;
int num_note;

constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

bool should_clear_buffer;
bool got_data;

static tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);

static tflite::MicroOpResolver<6> micro_op_resolver;

static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
tflite::MicroInterpreter* interpreter = &static_interpreter;

TfLiteTensor* model_input = interpreter->input(0);

int input_length;
TfLiteStatus setup_status = SetupAccelerometer(error_reporter);


void DisplaySongInfo(void);
void ModeSelection(void);
void Confirm(void);
void ChangeSong(void);
void GestureIdentify(void);
int PredictGesture(float* output);
void CallAudio(void);
void playNote(int freq);
void Stop(void);
int main()
{
    sw2.fall(queue.event(&Stop));
    Song = 0;
    t_DisplaySongInfo.start(callback(&q_DisplaySongInfo, &EventQueue::dispatch_forever));
    t_ModeSelection.start(callback(&q_ModeSelection, &EventQueue::dispatch_forever));
    t_DNN.start(callback(&q_DNN, &EventQueue::dispatch_forever));
    t_Confirm.start(callback(&q_Confirm, &EventQueue::dispatch_forever));
    t_ChangeSong.start(callback(&q_ChangeSong, &EventQueue::dispatch_forever));
    t.start(callback(&queue, &EventQueue::dispatch_forever));
    t_CallAudio.start(callback(&q_CallAudio, &EventQueue::dispatch_forever));
    ModeOrSong = 1;

  // Create an area of memory to use for input, output, and intermediate arrays.

  // The size of this will depend on the model you're using, and may need to be

  // determined by experimentation.

 // constexpr int kTensorArenaSize = 60 * 1024;

//  uint8_t tensor_arena[kTensorArenaSize];


  // Whether we should clear the buffer next time we fetch data




  should_clear_buffer = false;
  got_data = false;

  gesture_index = 2;

  // Set up logging.

//  static tflite::MicroErrorReporter micro_error_reporter;

//  tflite::ErrorReporter* error_reporter = &micro_error_reporter;


  // Map the model into a usable data structure. This doesn't involve any

  // copying or parsing, it's a very lightweight operation.

//  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);

//  model = tflite::GetModel(g_magic_wand_model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {

    error_reporter->Report(

        "Model provided is schema version %d not equal "

        "to supported version %d.",

        model->version(), TFLITE_SCHEMA_VERSION);

    return -1;

  }


  // Pull in only the operation implementations we need.

  // This relies on a complete list of all the ops needed by this graph.

  // An easier approach is to just use the AllOpsResolver, but this will

  // incur some penalty in code space for op implementations that are not

  // needed by this graph.

//  static tflite::MicroOpResolver<6> micro_op_resolver;

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                             tflite::ops::micro::Register_RESHAPE(), 1);

  micro_op_resolver.AddBuiltin(

      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,

      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,

                               tflite::ops::micro::Register_MAX_POOL_2D());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,

                               tflite::ops::micro::Register_CONV_2D());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,

                               tflite::ops::micro::Register_FULLY_CONNECTED());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,

                               tflite::ops::micro::Register_SOFTMAX());


  // Build an interpreter to run the model with

//  static tflite::MicroInterpreter static_interpreter(

//      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);

//  tflite::MicroInterpreter* interpreter = &static_interpreter;


  // Allocate memory from the tensor_arena for the model's tensors

  interpreter->AllocateTensors();


  // Obtain pointer to the model's input tensor

  // TfLiteTensor* model_input = interpreter->input(0);
  // model_input = interpreter->input(0);

  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||

      (model_input->dims->data[1] != config.seq_length) ||

      (model_input->dims->data[2] != kChannelNumber) ||

      (model_input->type != kTfLiteFloat32)) {

    error_reporter->Report("Bad input tensor parameters in model");

    return -1;

  }


//  int input_length = model_input->bytes / sizeof(float);
  input_length = model_input->bytes / sizeof(float);


//  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);


  if (setup_status != kTfLiteOk) {

    error_reporter->Report("Set up failed\n");

    return -1;

  }


 // error_reporter->Report("Set up successful...\n");
  uLCD.printf("Setup done!");


    sw2.rise(q_ModeSelection.event(ModeSelection));
    sw3.rise(q_Confirm.event(Confirm));
    /*
    if (ModeOrSong == 1) {
        sw2.rise(q_ModeSelection.event(ModeSelection));
    }
    if (ModeOrSong == -1) {
        sw2.rise(q_ChangeSong.event(ChangeSong));
    }
    sw3.rise(q_Confirm.event(Confirm));
    while (1) {
        q_DisplaySongInfo.event(DisplaySongInfo);
    }*/
}

void playNote(int freq)

{
  // uLCD.printf("In playNote\n");
 
  for(int i = 0; i < kAudioTxBufferSize; i++)

  {
    if (stop) {

      waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
    }
    else
    {
      waveform[i] = 0;
    }
    
  }

  audio.spk.play(waveform, kAudioTxBufferSize);

}

void Stop(void)
{
    stop = 0;
    ModeOrSong = 1;
    for(int i = 0; i < kAudioTxBufferSize; i++)

    {
    
        waveform[i] = 0;
    
    }

    audio.spk.play(waveform, kAudioTxBufferSize);
    q_ModeSelection.call(ModeSelection);
}

void CallAudio(void)

{
  //uLCD.printf("In CallAudio\n");

  //t.start(callback(&queue, &EventQueue::dispatch_forever));
  int i = 0;

  serialCount = 0;

  while(serialCount < 3) {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
  }
  serialInBuffer[serialCount] = '\0';
  num_note = (int) atoi(serialInBuffer);
  //uLCD.printf("Number of note: %d", num_note);
  serialCount = 0;


  while(i < num_note)

  {

    if(pc.readable())

    {

      serialInBuffer[serialCount] = pc.getc();

      serialCount++;

      if(serialCount == 3)

      {

        serialInBuffer[serialCount] = '\0';

        song[i] = (int) atoi(serialInBuffer);

        serialCount = 0;

        //uLCD.printf("song = %d\n", song[i]);

        i++;

      }

    }

  }
  i = 0;

  serialCount = 0;

  while(i < num_note)

  {

    if(pc.readable())

    {

      serialInBuffer[serialCount] = pc.getc();

      serialCount++;

      if(serialCount == 3)

      {

        serialInBuffer[serialCount] = '\0';

        noteLength[i] = (int) atoi(serialInBuffer);

        serialCount = 0;

        //uLCD.printf("noteLength = %d\n", noteLength[i]);

        i++;

      }

    }

  }

  for(int i = 0; i < (num_note) && (stop); i++)

  {

    int length = noteLength[i];

    while((length--)  && (stop))

    {

      // the loop below will play the note for the duration of 1s

      for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)

      {

        queue.call(playNote, song[i]);

      }

      if(length < 1) wait(1.0);
      if (!stop) break;

    }

  }
  // uLCD.printf("End of CallAudio\n");
}

void Confirm(void)
{
   //uLCD.printf("In confirm\n");
    if (ModeOrSong == -2) {
      ModeOrSong = -1;
      DisplaySongInfo();
    }
    else {
        ModeOrSong--;
        // uLCD.cls();
        // uLCD.printf("Mode/Song = %d", ModeOrSong);
        if (ModeOrSong == 0) {
            //uLCD.printf("Mode = %d", Mode);
            if (Mode == 0) {
                Song--;
                ModeOrSong = -1;
                if (Song == -1) {
                    Song = 2;
                }
                //uLCD.printf("Song = %d", Song);
                DisplaySongInfo();
            }
            else if (Mode == 1) {
                ModeOrSong = -1;
                Song++;
                if (Song == 3) {
                    Song = 0;
                }
                //uLCD.printf("Song = %d", Song);
                DisplaySongInfo();
            }
            else {
                //uLCD.printf("Song = %d", Song);
                ModeOrSong = -2;
                // uLCD.cls();
                //uLCD.printf("ModeOrSong = %d\n", ModeOrSong);
                q_ModeSelection.call(ModeSelection);
                //ModeSelection();
              }
          }
    }
    /*
    if (ModeOrSong == -2) {
        ModeOrSong = 1;

    }*/
}

void ChangeSong(void)
{
  // uLCD.cls();
  // uLCD.printf("ModeOrSong = %d\n", ModeOrSong);
  DisplaySongInfo();
  while(ModeOrSong == -2) {
    //uLCD.printf("In ChangeSong\n");
    gesture_index_pre = gesture_index;
    //uLCD.printf("%d, %d\n", gesture_index, gesture_index_pre);
    GestureIdentify();
    //uLCD.printf("%d, %d\n", gesture_index, gesture_index_pre);
    Song_pre = Song;
    // printf("%d\r\n", gesture_index);
    if (gesture_index == 0) {
        Song--;
        if (Song == -1) {
            Song = 2;
        }
        DisplaySongInfo();
    }
    else if (gesture_index == 1) {
        Song++;
        if (Song == 3) {
            Song = 0;
        }
        DisplaySongInfo();
    }
  }
  
}

void ModeSelection(void)
{
    //uLCD.printf("In ModeSelection\n");
    if (ModeOrSong == -2) {
      ChangeSong();
    }
    //ModeOrSong = 1;
    

    // GestureIdentify();
    // q_DNN.event(GestureIdentify);
    // q_DisplaySongInfo.event(DisplaySongInfo);
  //  uLCD.printf("Before while\n");
  if (ModeOrSong == 1){
    DisplaySongInfo();
    while (ModeOrSong == 1) {
        gesture_index_pre = gesture_index;
        //uLCD.printf("%d, %d\n", gesture_index, gesture_index_pre);
        GestureIdentify();
        //uLCD.printf("%d, %d\n", gesture_index, gesture_index_pre);
        Mode_pre = Mode;
        // printf("%d\r\n", gesture_index);
        if (gesture_index == 0) {
            Mode--;
            if (Mode == -1) {
                Mode = 2;
            }
            DisplaySongInfo();
        }
        else if (gesture_index == 1) {
            Mode++;
            if (Mode == 3) {
                Mode = 0;
            }
            DisplaySongInfo();
        }
        /*
        if (gesture_index_pre != gesture_index) {
            Mode_pre = Mode;
            if (gesture_index == 0) {
                Mode--;
                if (Mode == -1) {
                    Mode = 2;
                }
            }
            else if (gesture_index == 1) {
                Mode++;
                if (Mode == 3) {
                    Mode = 0;
                }
            }*/
            //DisplaySongInfo();
            //uLCD.printf("Mode = %d", Mode);
    }
  }
}
void DisplaySongInfo()
{
    // uLCD.printf("In DisplaySongInfo\n");
    if (ModeOrSong == 1) {
        if (Mode == 0) {
            uLCD.cls();
            //uLCD.set_font(FONT_5X5);
            uLCD.color(GREEN);
            // uLCD.text_width(1.5);
            // uLCD.text_height(1.5);
            uLCD.printf("Forword\n");
            //uLCD.set_font(FONT_2X2);
            // uLCD.text_width(1);
            // uLCD.text_height(1);
            uLCD.color(BLUE);
            uLCD.printf("backword\n");
            uLCD.printf("Song Selection\n");
            wait(0.5);
        }
        else if (Mode == 1) {
            uLCD.cls();
            //uLCD.set_font(FONT_2X2);
            // uLCD.text_width(1);
            // uLCD.text_height(1);
            uLCD.color(BLUE);
            uLCD.printf("Forword\n");
            //uLCD.set_font(FONT_5X5);
            // uLCD.text_width(1.5);
            // uLCD.text_height(1.5);
            uLCD.color(GREEN);
            uLCD.printf("backword\n");
            //uLCD.set_font(FONT_2X2);
            // uLCD.text_width(1);
            // uLCD.text_height(1);
            uLCD.color(BLUE);
            uLCD.printf("Song Selection\n");
            wait(0.5);
        }
        else {
            uLCD.cls();
            //uLCD.set_font(FONT_2X2);
            // uLCD.text_width(1);
            // uLCD.text_height(1);
            uLCD.color(BLUE);
            uLCD.printf("Forword\n");
            uLCD.printf("backword\n");
            //uLCD.set_font(FONT_5X5);
            //uLCD.text_width(1.5);
            // uLCD.text_height(1.5);
            uLCD.color(GREEN);
            uLCD.printf("Song Selection\n");
            //uLCD.set_font(FONT_2X2);
            // uLCD.text_width(1);
            // uLCD.text_height(1);
            wait(0.5);
        }
    }
    else if (ModeOrSong == -2) {
      if (Song == 0) {
        uLCD.cls();
        uLCD.color(GREEN);
        //uLCD.set_font(FONT_5X5);
        uLCD.text_width(1.5);
        uLCD.text_height(1.5);
        uLCD.printf("Twinkle twinkle little star\n"); //Default Green on black text
        uLCD.color(BLUE);
        //uLCD.set_font(FONT_2X2);
        uLCD.text_width(1);
        uLCD.text_height(1);
        uLCD.printf("B\n"); //Default Green on black text
        uLCD.printf("C\n"); //Default Green on black text
        wait(0.5);
      }
      else if (Song == 1) {
        uLCD.cls();
        uLCD.color(BLUE);
        //uLCD.set_font(FONT_2X2);
        uLCD.text_width(1);
        uLCD.text_height(1);
        uLCD.printf("Twinkle twinkle little star\n"); //Default Green on black text
        uLCD.color(GREEN);
        //uLCD.set_font(FONT_5X5);
        uLCD.text_width(1.5);
        uLCD.text_height(1.5);
        uLCD.printf("B\n"); //Default Green on black text
        uLCD.color(BLUE);
        //uLCD.set_font(FONT_2X2);
        uLCD.text_width(1);
        uLCD.text_height(1);
        uLCD.printf("C\n"); //Default Green on black text
        wait(0.5);
      }
      else if (Song == 2) {
        uLCD.cls();
        uLCD.color(BLUE);
        //uLCD.set_font(FONT_2X2);
        uLCD.text_width(1);
        uLCD.text_height(1);
        uLCD.printf("Twinkle twinkle little star\n"); //Default Green on black text
        uLCD.printf("B\n"); //Default Green on black text
        uLCD.color(GREEN);
        //uLCD.set_font(FONT_5X5);
        uLCD.text_width(1.5);
        uLCD.text_height(1.5);
        uLCD.printf("C\n"); //Default Green on black text
        wait(0.5);
      }
    }
    else if (ModeOrSong == -1) {
      if (Song == 0) {
        uLCD.cls();
        uLCD.color(GREEN);
        uLCD.printf("Twinkle twinkle little star is playing."); //Default Green on black text
        uLCD.triangle(35, 35, 35, 95, 95, 65, GREEN);
        pc.printf("0\r\n");
        stop = 1;
        q_CallAudio.call(CallAudio);
        //ModeOrSong = 1;
        //q_ModeSelection.call(ModeSelection);
        //wait(0.5);
      }
      else if (Song == 1) {
        uLCD.cls();
        uLCD.color(GREEN);
        uLCD.printf("B is playing."); //Default Green on black text
        uLCD.triangle(35, 35, 35, 95, 95, 65, GREEN);
        pc.printf("1\r\n");
        stop = 1;
        q_CallAudio.call(CallAudio);
        //ModeOrSong = 1;
        //q_ModeSelection.call(ModeSelection);
        //wait(0.5);
      }
      else if (Song == 2) {
        uLCD.cls();
        uLCD.color(GREEN);
        uLCD.printf("C is playing."); //Default Green on black text
        uLCD.triangle(35, 35, 35, 95, 95, 65, GREEN);
        pc.printf("2\r\n");
        stop = 1;
        q_CallAudio.call(CallAudio);
        //ModeOrSong = 1;
        //q_ModeSelection.call(ModeSelection);
        //wait(0.5);
      }
      //uLCD.printf("ModeOrSong = %d\n", ModeOrSong);
    }
}
// Return the result of the last prediction

int PredictGesture(float* output) {

  // How many times the most recent gesture has been matched in a row

  static int continuous_count = 0;

  // The result of the last prediction

  static int last_predict = -1;


  // Find whichever output has a probability > 0.8 (they sum to 1)

  int this_predict = -1;

  for (int i = 0; i < label_num; i++) {

    if (output[i] > 0.8) this_predict = i;

  }


  // No gesture was detected above the threshold

  if (this_predict == -1) {

    continuous_count = 0;

    last_predict = label_num;

    return label_num;

  }


  if (last_predict == this_predict) {

    continuous_count += 1;

  } else {

    continuous_count = 0;

  }

  last_predict = this_predict;


  // If we haven't yet had enough consecutive matches for this gesture,

  // report a negative result

  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {

    return label_num;

  }

  // Otherwise, we've seen a positive result, so clear all our variables

  // and report it

  continuous_count = 0;

  last_predict = -1;


  return this_predict;

}


void GestureIdentify(void) {
  //uLCD.printf("In GestureIdentify\n");
  got_data = false;

 while ((ModeOrSong == 1) || (ModeOrSong == -2)) {
    /*if (ModeOrSong == -2) {
        uLCD.printf("In GestureIdentify\n");
    }
*/

      // Attempt to read new data from the accelerometer

    got_data = ReadAccelerometer(error_reporter, model_input->data.f,

                                 input_length, should_clear_buffer);


    // If there was no new data,

    // don't try to clear the buffer again and wait until next time

    if (!got_data) {

      should_clear_buffer = false;

      continue;

    }


    // Run inference, and report any error

    TfLiteStatus invoke_status = interpreter->Invoke();

    if (invoke_status != kTfLiteOk) {

      error_reporter->Report("Invoke failed on index: %d\n", begin_index);

      continue;

    }


    // Analyze the results to obtain a prediction

    gesture_index = PredictGesture(interpreter->output(0)->data.f);


    // Clear the buffer next time we read data

    should_clear_buffer = gesture_index < label_num;


    // Produce an output

    if (gesture_index < label_num) {

      // error_reporter->Report(config.output_message[gesture_index]);
      break;
    }

    //uLCD.printf("End of GestureIdentify\n");

  }

}