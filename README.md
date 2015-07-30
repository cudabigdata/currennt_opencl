=================UPDATE==================
7/30/2015 : Support for OpenCL added
=========================================

CURRENNT is written in C++/CUDA and runs on Windows and Linux (Which now support both OpenCL for all Nvidia, AMD, Intel GPU cards). It requires a
CUDA-capable graphics card with a compute capability of 1.3 or higher (i.e. at
least a GeForce 210, GT 220/40, FX380 LP, 1800M, 370/380M or NVS 2/3100M)

This is free software licensed under the GPL; see LICENSE for details.
If you find CURRENNT useful for your research, we would appreciate if you
cite the following paper,

Felix Weninger, Johannes Bergmann, and Bjoern Schuller,
"Introducing CURRENNT - the Munich Open-Source CUDA RecurREnt Neural Network
Toolkit", Journal of Machine Learning Research, 2014.

+=============================================================================+
| Structure                                                                   |
+=============================================================================+
1. Building on Linux
2. Building on Windows
3. Command line options
    3.1 Common options
    3.2 Feed forward options
    3.3 Training options
    3.4 Autosave options
    3.5 Data file options
    3.6 Weight initialization options
4. Network configuration
5. NetCDF data files
    5.1 General structure
    5.2 Regression tasks
    5.3 Classification tasks
6. Tests
7. Examples

+=============================================================================+
| 1. Building on Linux                                                        |
+=============================================================================+
Building on Linux requires the following:
* CUDA Toolkit 5.0 
* GCC 4.6 or higher
* NetCDF scientific data library
* Boost library 1.48 or higher

To build CURRENNT execute the following commands:
#> cd currennt
#> mkdir build && cd build
#> cmake ..
#> make

This produces the executable file 'currennt'.

You might get many warnings like 'Cannot tell what pointer points to, assuming
global memory space'. These are totally OK and there seems to be no way to
suppress them.


+=============================================================================+
| 2. Building on Windows                                                      |
+=============================================================================+
Building on Windows requires the following:
* Visual Studio 2010
* CUDA Toolkit 5.0
* Boost library 1.52.0 or higher

The NetCDF sources and required DLLs are already there, so you do not have to
download them yourself.

To build CURRENNT do the following:
1) Open 'currennt.sln' in Visual Studio 2010
2) Ensure that you installed the Boost library in 'C:/boost_1_52_0'. 
   NOTE: If you build Boost from source, make sure you build the x64 version
   (default is x86). This is done via
     bjam msvc architecture=x86 address-model=64 stage
   on the command line from the Boost source directory.
   If you want to use another version or another installation path, you need 
   to modify  the project files for the 'currennt_lib' and 'currennt' projects. 
   You can do this by changing the include and library search paths under the
   project settings or you can edit the project files directly which is much
   simpler. To do this, open the files 'currennt_lib/currennt_lib.vcxproj'
   and 'currennt/currennt.vcxproj' and replace every occurence of
   'C:/boost_1_52_0' with the path (may be a relative path) to your Boost
   installation.
3) Ensure that the solution configuration is set to 'Release' on 'x64' (in the
   toolbar right below the main menu)
4) Compile the whole solution via 'Build -> Build Solution' (F7)

You might get many warnings like 'Cannot tell what pointer points to, assuming
global memory space'. These are totally OK and there seems to be no way to
suppress them.

The build process creates currennt.exe in the Release directory.
If you have not installed the NetCDF library and its dependencies in your
global system path, you have to copy them from the CURRENNT source directory 
into the Release directory.


+=============================================================================+
| 3. Command line options                                                     |
+=============================================================================+
CURRENNT offers many command line options. You can get a short description of
them by running the program with the '--help' switch. A more detailed
description of the options can be found in the following subsections.
All command line options can also be specified in a parameter file 
(cf. --options_file below).

+-----------------------------------------------------------------------------+
| 3.1 Common options                                                          |
+-----------------------------------------------------------------------------+
--help
  Shows a brief description of every available option.
  
--options_file <file.cfg>
  This option reads the command line options from <file.cfg> in which every
  option is set in a separate line in the form of 'option = value'. This is a
  positional option, i.e. you can omit the name of the option and just run the
  trainer as 'currennt mynet.cfg'. Go into the directory 'examples' for an
  example of this feature. Options set on the command line have priority, i.e.
  if you set 'max_epochs = 5' in <file.cfg> and set 'max_epochs = 10' on the
  command line, the final value of this option will be 10.
  
--network <file.jsn>
  Sets the file which defines the structure of the neural network. If the file
  contains weights, these weights are used instead of initializing them using a
  random distribution. The default value of this option is 'network.jsn'. The
  structure of this file is explained in section 4.
  
--cuda <true/false>
  Enables CUDA. If you explicitly set this option to 'false' then the CPU will
  be used for all computations. This only makes sense for debugging or for
  training very small networks (<< 100,000 weights). CUDA is enabled by
  default. The device that is being used is specified by the CURRENNT_CUDA_DEVICE
  environment variable (default: 0).
  
--list_devices <true/false>
  If yes, instead of doing any processing, the list of available CUDA devices
  along with their IDs is printed and the program exits. To change the CUDA
  device being used, set the CURRENNT_CUDA_DEVICE environment variable, e.g.,
  on bash: $ export CURRENNT_CUDA_DEVICE=<device_id>

--parallel_sequences <value>
  In order to speed up computations, the trainer processes a set of multiple
  training sequences, which we call a fraction, in parallel. I.e. if you set
  this value to 50 then the trainer will split the whole data set in fractions
  of 50 sequences each and calculates all those sequences concurrently. This
  is the ultimate option to boost the training speed but you should be careful
  because choosing a value != 1 means that you do not do true online learning
  any more if that's important for your task. If you can afford it, choose a
  value as high as possible (if you go too high the program tells you that
  there is not enough GPU memory available). This can speedup the training over
  20x compared to true online learning! Section 3.3 contains more information
  about this option.
  
--random_seed <value>
  Sets the seed for the random number generators. This option allows you to
  re-produce training results on the same machine. Useful for debugging but
  usually you should leave it at its default value of 0 which causes the
  program to create the seed itself. 

+-----------------------------------------------------------------------------+
| 3.2 Forward pass options                                                    |
+-----------------------------------------------------------------------------+
In forward pass mode, the trainer does not train a network but uses an
already trained network (specified by the --network option) and computes only
the forward pass to obtain the network outputs for certain input sequences.

--ff_input_file <file.nc>
  Sets the name of the data file (in NetCDF format, see section 5) that
  contains the network input sequences for the forward pass. Data files
  used in forward pass must have the same structure as training data files
  (see section 5), but the vectors of outputs or target classes may be missing.
  If you set static noise (via '--input_noise_sigma', see below) != 0, then the
  noise is also applied to the input sequences of this file. If you don't want
  that, ensure that static noise is set to 0.
  
--ff_output_file <file>
  Sets the name of the output file that is produced in forward pass. 
  The interpretation of this option depends on the chosen output format (cf.
  --ff_output_format)
  If the output format is "single CSV", it specifies the name of a CSV file;
  otherwise, it specifies the name of a directory where for each sequence,
  a file with the output activations will be created.
  The default output file name is 'ff_output.csv'.

--ff_output_format <format>
  Sets the output format for the forward pass (htk, csv, or single_csv).

  In "single_csv" mode, a single CSV file is written
  in which the first column contains the sequence tag from
  the input data file and the other columns contain the activations of the
  output neurons in temporally ascending order. I.e. if your network has an
  output layer with 2 neurons in it and the input data file contains two
  sequences with tags A and B whereas A has a length of 2 and B has a length of
  3 timesteps, then the resulting output file could look like
    A;0.1;0.2;-0.05;0.9723
    B;0.5;0.5;0.111;-0.22;0.82;0.3
  
  In "csv" mode, every output sequence is written to a CSV file whose name
  corresponds to the sequence tag plus ".csv". 
  If the sequence tag can be interpreted as a
  path name on your system, subdirectories will be created accordingly.
  For example, if there are two sequences called "speaker1/sentence1" and
  "speaker2/sentence3", the directories speaker1 and speaker2 will be created,
  and the CSV output will be written to the sentence1.csv and sentence3.csv files
  within these directories.

  In "htk" mode, the logic is similar, but instead of CSV files, binary files
  in the Hidden Markov Toolkit (HTK) feature file format will be output.
  If you don't know what HTK is, you will probably not need this option.

  The default output format is "single_csv".

--ff_output_kind <number>
  Sets the feature type "magic number" in the HTK feature files. 
  Only relevant if ff_output_format = htk.
  Default is 9 ("user defined").

--feature_period <number>
  Sets the sampling period of the features in the HTK feature files, given in
  milliseconds.  Default is 10.

--revert_std <true/false>
  If set to true, it is attempted to read the outputMeans and outputStdevs
  fields from the forward pass input file, which are interpreted as the
  original means and standard deviations of the output features, while it is
  assumed that the output features are standardized.  Then, the output features
  are multiplied by their standard deviations and the means are added.  If the
  fields cannot be read, no operation is performed.  This is only valid for
  regression tasks. Default is "true".  See the nc-standardize tool in the
  tools folder for creating standardized NetCDF files.

+-----------------------------------------------------------------------------+
| 3.3 Training options                                                        |
+-----------------------------------------------------------------------------+
--train <true/false>
  Enables training mode. If you want to train a network then you have have to
  set this option to 'true'. The default value is 'false' and hence the program
  does only a forward pass by default.

--stochastic <true/false>
  Enables stochastic gradient descent. If you set this value to 'true'
  then the trainer updates the network weights after every processed fraction
  (which consists of PS sequences, with PS being the number of parallel
  sequences set via the --parallel_sequences option). If you set this value to
  'false' which is the default value, then you're doing batch learning. If you
  want true online learning (weight updates after every sequence) then you have
  to set this value to 'true' and set '--parallel_sequences' to 1.

--hybrid_online_batch <true/false>
  Does the same, provided for compatibility.

--shuffle_fractions <true/false>
  Enables shuffling of fractions during network training. Since only the
  fractions are shuffled, they always consist of the same sequences. The
  advantage of fraction shuffling over sequence shuffling (see below) is
  that it can be considerably faster if the lengths of your input sequences
  differ greatly. If you have roughly equally long sequences, prefer sequence
  shuffling. The default value is 'false'.

--shuffle_sequences <true/false>
  Enables shuffling of sequences within and across fractions. As opposed to
  fraction shuffling (see above), this mode truely randomizes the order of all
  sequences in each training epoch. If the lengths of your input sequences
  differ greatly, you can use fraction shuffling instead which should speed up
  the training at the cost of less randomized sequences.

--max_epochs <value>
  Sets the maximum number of training epochs. If the network does not converge
  after <value> epochs, the training is stopped, no matter what. The default
  value of this option is infinity.

--max_epochs_no_best <value>
  Causes the program to stop training if no new best error on the validation
  set could be achieved within the last <value> epochs. Be careful if you also
  use the '--validate_every' option (see below). If you choose to only
  calculate the validation error every 5 epochs, then the trainer might miss
  new best errors in epochs in which this error is not evaluated. The default
  value of this option is 20.

--validate_every <value>
  Causes the trainer to evaluate the validation error every <value> epochs.
  Choosing a value other than 1 speeds up training times but you might miss new
  best errors. The default value is 1.

--test_every <value>
  Causes the trainer to evaluate the error on the test set every <value>
  epochs. The default value is 1.

--optimizer steepest_descent
  Sets the type of optimizer to use. The default (and currently only) optimizer
  is a steepest descent optimizer with momentum ('steepest_descent'). Its
  parameters can be set via the options '--learning_rate' and '--momentum' (see
  below).

--learning_rate <value>
  Sets the learning rate for the steepest descent optimizer. The default value
  is 1e-5.

--momentum <value>
  Sets the momentum for the steepest descent optimizer. The default value is
  0.9.

--weight_noise_sigma <value>
  Sets the standard deviation of the Gaussian noise that is applied to weights
  before calculating the gradient of each batch. This may provide more
  robustness against overfitting.  The default value is 0.0 (no noise).

--save_network <file.jsn>
  Sets the file name of the network file that is created when training is
  finished. The file will contain the network structure (the same as the one
  loaded from the file provided by '--network') and the trained weights. You
  can use this file to re-train the network by simply using it as value for
  the '--network' option. The trainer will then use these weights as initial
  weights instead of initializing them randomly.

+-----------------------------------------------------------------------------+
| 3.4 Autosave options                                                        |
+-----------------------------------------------------------------------------+
CURRENNT offers options to save to current training status after every
training epoch. Autosave is disabled by default but if you're training on a
large amount of data, you should really switch it on. If you don't and your
computer crashes during training, you have to start all over again.

The produced autosave files are in JSON format and can be used as network file
for the '--network' option. This means you can first train the network a few
epochs with configuration A and then continue training with a completely 
different configuration B by restarting the program with the autosave file.

WARNING: If an autosave file already exists, it will be overwritten!

--autosave <true/false>
  Enables autosave after every training epoch. The resulting files are named
  '<prefix>epoch0123.autosave' with <prefix> being the prefix configured via
  '--autosave_prefix'. Default is 'false'. Note that using this option is
  time-consuming (as networks have to be serialized after every epoch) and
  requires a lot of disk space if you have large networks!
  
--autosave_best <true/false>
  Enables autosave once a new best validation error is reached.
  The resulting files will be written to <autosave_prefix>.best.jsn.
  Using this option is recommended if training your network takes significant
  time.  Default is 'false'.

--autosave_prefix <value>
  Sets the filename prefix for autosave files. If you want your autosave files
  to be put in the directory 'mydir' and have filenames like
  'mynn-epoch012.autosave', then you should set <value> to 'mydir/mynn-'.
  The default prefix is empty, so the resulting files are put in the current
  working directory with names like 'epoch012.autosave'
  
--continue <file.autosave>
  Continues training from the provided autosave file. If you continue from an
  autosave file, all options set at the command line or in the options file are
  ignored as they are stored in the autosave file itself. If you really want to
  change these options, you can edit the autosave file since it is an ordinary
  JSON file. WARNING: If you do this (a) you have to be careful not to break
  anything (special characters, delimiters, ...) and (b) you lose the
  information about which options were used to obtain the autosave file in the
  first place. To continue training, you only need the autosave file and the
  training/validation/test data files.

+-----------------------------------------------------------------------------+
| 3.5 Data file options                                                       |
+-----------------------------------------------------------------------------+
--train_file <file.nc[,file2.nc...]>
  Sets the NetCDF file(s) which contain(s) the sequences used as the training
  set during network training.  Separate multiple files by semicolons or
  commas.  The required structure of the data file is described in section 5.
  Does not have a default value and must be set in training mode.

--val_file <file.nc[,file2.nc...]>
  Sets the NetCDF file(s) which contain(s) the sequences used as the validation
  set during network training.  Separate multiple files by semicolons or
  commas.  The required structure of the data file is described in section 5.
  Does not have a default value. If you don't provide a validation file, then
  training is done without it.

--test_file <file.nc[,file2.nc...]>
  Sets the NetCDF file(s) which contain(s) the sequences used as the test set
  during network training.  Separate multiple files by semicolons or commas.
  The required structure of the data file is described in section 5. Does not
  have a default value. If you don't provide a test file, then training is done
  without it.

--train_fraction <value>
  Sets the fraction of the training set that will be used. I.e. if you set this
  value to 0.5, then only the first half of the sequences from the file set via
  '--train_file' will be used.

--val_fraction <value>
  Sets the fraction of the validation set that will be used. I.e. if you set
  this value to 0.5, then only the first half of the sequences from the file
  set via '--val_file' will be used.

--test_fraction <value>
  Sets the fraction of the test set that will be used. I.e. if you set this
  value to 0.5, then only the first half of the sequences from the file set via
  '--test_file' will be used.

--truncate_seq <value>
  Truncates over-long training sequences to a maximum length.
  This can greatly speed up training if your sequences vary much in length,
  and enable training with very long sequences that might not fit into the GPU
  memory.  If a sequence is longer than <truncate_seq> * 1.5 time steps, the
  remaining time steps will be assigned to a new sequence, and this process
  will be continued iteratively until no sequence is longer than <truncate_seq>
  * 1.5 time steps. As a result, no sequence will be shorter than 0.5 *
  * <truncate_seq> time steps.  Default is 0 (no truncation).

--input_noise_sigma <value>
  Sets the standard deviation of the static noise that is applied to the input
  sequences of the training and feed forward input set. Static noise is not
  applied to validation and test sets. The default value is 0.0 (no static
  noise)

--input_left_context <value>
--input_right_context <value>
  Enables frame splicing, i.e., concatenating multiple feature frames from a
  sliding window.  The features which are effectively used at time step t span
  the features from timesteps t-<input_left_context> through
  t+<input_right_context>.  The first and last frames are duplicated as
  necessary.  Default is no frame splicing, i.e., input_left_context =
  input_right_context = 0.

--output_time_lag <value>
  Enables training with output time lag. This is mainly useful for having
  unidirectional RNNs with lookahead.  For example, if this is set to 5, the
  network will be trained to predict the given targets 5 time steps before.  If
  this is given in forward pass mode, the outputs are shifted to the "left"
  by <output_time_lag> frames, and the last output time step is duplicated as
  necessary.  Default is no time lag.

--cache_path <string>
  Sets the path for caching data (default /tmp).

+-----------------------------------------------------------------------------+
| 3.6 Weight initialization options                                           |
+-----------------------------------------------------------------------------+
The network weights are initialized randomly unless specified in the network
file set via the '--network' option. The random weights are drawn from either
a uniform or a normal distribution.

--weights_dist <uniform/normal>
  Sets the distribution to use for initializing the weights. Default is
  'uniform'.
  
--weights_uniform_min <value>
  Sets the minimum value for weights when using the uniform distribution.
  Default is -0.1.

--weights_uniform_max <value>
  Sets the maximum value for weights when using the uniform distribution.
  Default is 0.1.
  
--weights_normal_sigma <value>
  Sets the standard deviation for the normal distribution used to initialize
  the weights. Default is 0.1.

--weights_normal_mean <value>
  Sets the standard deviation for the normal distribution used to initialize
  the weights. Default is 0.
  
 
+=============================================================================+
| 4. Network configuration                                                    |
+=============================================================================+
The structure of a neural network is defined in a JSON file and passed to the
CURRENNT executable via the '--network' option. The structure of such files
is described in this chapter.

As an example, we will create a neural network for multiclass classification
tasks. Our training data consists of input sequences with patterns of 39 values
and target sequences with integers denoting 1 out of 51 target classes for each
timestep. Hence, we need a neural network with 39 input neurons and 51 output
neurons. Each of the output neurons shall represent the probability of one
timestep belonging to the corresponding target class. Hence, the sum over all
output neuron activations shall equal 1.

Our network shall contain a single hidden LSTM layer that is trained in both
positive and negative time directions (-> bidirectional LSTM). The hidden layer
shall contain 100 biased LSTM units. The final network file has the following
content:

{
    "layers": [
        {
            "size": 39,
            "name": "input_layer",
            "type": "input"
        },
		{
            "size": 100,
            "name": "hidden_layer",
            "bias": 1.0,
            "type": "blstm"
        },
        {
            "size": 51,
            "name": "output_layer",
            "bias": 1.0,
            "type": "softmax"
        },
        {
            "size": 51,
            "name": "postoutput_layer",
            "type": "multiclass_classification"
        }
    ]
}

It is very important to use the right braces and commas in the right places. If
the syntax is not 100% correct, CURRENNT will not accept the file. It is also
crucial to define the layers in the right order. The first layer must always be
the input layer (with type 'input') and the last layer must always be a post
output layer.

The so-called post output layer is always the very last layer in the network.
It does not have any trainable weights and its purpose is to evaluate the 
objective function during the forward pass and induce the error into the output
layer during the backward pass.

In this example, our network consists of an input layer with 39 neurons, a
hidden bidirectional LSTM layer, a feed forward output layer with 51 neurons
which use the softmax activation function and the post output layer suitable
for our multiclass classification task.

You can define an arbitrary number of hidden layers with different types and
sizes. See the directory 'examples' for more example network files.

Some important points:
* Input and post output layers do not require a bias value.
* You must provide a bias value for hidden and output layers. If you do not
  want to have bias connections, set the bias to 0.
* Layer names must be unique
* The post output layer has the same size as the output layer, except
  for the sse_mask type (cf. below)
* The size of the post output layer must match the number of targets in the
  NetCDF file in case of regression tasks, and must be equal to the number of
  classes for multi-way classification tasks.
* It is possible to specify learning rates per layer, by adding
   "learningRate": <learning_rate>
  as key-value pair in the JSON element corresponding to the desired layer.
  For example, this can be used to manually define transformations of features
  as "deterministic", i.e. non-trainable, layers, or to prevent pre-trained
  layers from being updated, by setting the learning rate to zero.

Available hidden and output layer types:
* feedforward_tanh     = Feed forward layer with tanh as activation function
* feedforward_logistic = Feed forward layer with a logistic activation function
* feedforward_identity = Feed forward layer with f(x)=x activation function
* softmax              = Feed forward layer with softmax activation function
* lstm                 = Unidir. LSTM layer with forget gates and peepholes
* blstm                = Bidir.  LSTM layer with forget gates and peepholes

Available post output layers: 
("N targets" refers to the targetPatternSize in the NetCDF file for regression
and "N classes" to the numLabels field for classification)

* sse: Sum of Squared Error objective function
  To be used with any output layer. Requires N targets for output layer of size
  N

* weighted_sse: Weighted Sum of Squared Error objective function
  To be used with any output layer. Requires 2N targets for output layer of
  size N. The targets and the weights are expected to be interleaved, i.e., any
  vector of targets should be of the form (t1, w1, t2, w2, ..., tN, wN).

* rmse: Root Mean Squared Error objective function
  To be used with any output layer. Requires N targets for output layer of size
  N

* ce: Cross entropy objective function 
  To be used with a softmax output layer. Requires N targets for output layer
  of size N. Note: If you want to train multi-way classification, it is much
  more efficient to use multiclass_classification, which avoids explicit
  storage of all training targets (most of which are zero); this is only useful
  if you want to train non-trivial discrete PDFs.

* binary_classification: Cross entropy for binary classification
  Requires 1 class and a feedforward_logistic output layer of size 1

* multiclass_classification: Cross entropy objective function for 1-of-C coding
  of C-way classification targets
  Requires C classes and a softmax output layer of size C.

* sse_mask: Sum of Squared Error objective function for masking output
  Requires 2N targets, and a feedforward_logistic output layer of size N.  The
  output layer activations are interpreted as [0,1] mask for a sequence of data
  vectors (e.g., a noisy speech spectrogram) so that the product of the mask
  computed by the network and the masking input should yield the desired
  masking output (e.g., a clean speech spectrogram).  This can be used, e.g.,
  to train audio source separation.  The masking outputs and masking inputs are
  given alternatingly, i.e., the target vector in the NetCDF file is supposed
  to be of the form (o1, i1, o2, i2, ..., oN, iN) where o denotes masking
  output and i denotes masking input.  Note that the "masking inputs" need not
  correspond to the features at the input layer, e.g., the input layer can read
  log-Mel spectra while the input data is a magnitude spectrogram.  This is the
  configuration used for the experiments in (Weninger et al., Discriminatively
  trained recurrent neural networks for single-channel speech separation, IEEE
  GlobalSIP, 2014).


+=============================================================================+
| 5. NetCDF data files                                                        |
+=============================================================================+
The data files used for CURRENNT are NetCDF (*.nc) files. An example is
provided in the directory 'examples' and its structure can be investigated by
running 'ncdump nc_file.nc'.

The following subsections describe the general structure of these files and the
required extensions for regression and classification tasks respectively.

+-----------------------------------------------------------------------------+
| 5.1 General structure                                                       |
+-----------------------------------------------------------------------------+
The data files need to contain the following dimensions:
* numSeqs         = Number of sequences
* numTimesteps    = Total number of timesteps
* inputPattSize   = Size of each input pattern (= number of input neurons)
* maxSeqTagLength = Maximum length of a sequence tag

The required variables are:
* char seqTags(numSeqs, maxSeqTagLength)    = Tag (name) for each sequence
* int seqLengths(numSeqs)                   = Length of each sequence
* float inputs(numTimesteps, inputPattSize) = Input Patterns

+-----------------------------------------------------------------------------+
| 5.2 Regression tasks                                                        |
+-----------------------------------------------------------------------------+
Additional dimensions:
* targetPattSize = Size of each output pattern (= number of output neurons)

Additional variables:
* float targetPatterns(numTimesteps, targetPattSize) = Target patterns

Optional variables:
* float outputMeans(targetPattSize) = estimated means of outputs
* float outputStdevs(targetPattSize) = estimated standard deviations of outputs
  (used to revert standardization of outputs in forward pass mode)

+-----------------------------------------------------------------------------+
| 5.3 Classification tasks                                                    |
+-----------------------------------------------------------------------------+
Additional dimensions:
* numLabels = Number of target classes

Additional variables:
* int targetClasses(numTimesteps) = Target classes (one for each timestep)


+=============================================================================+
| 6. Tests                                                                    |
+=============================================================================+
The directory 'tests' contains the scripts which check if the library produces
the correct output for a predefined input. Currently there is only one test
'test1' which traines a small deep RNN for one epoch and compares the trained
weights with reference weights obtained with RNNLIB. To run the test, execute
the 'run.py' Python script from within the directory 'test1'. The result will
be printed on the screen. Make sure that you have built CURRENNT as described
above before you run a test.


+=============================================================================+
| 7. Examples                                                                 |
+=============================================================================+
The directory 'examples' contains example configurations that show how to use
CURRENNT for classification and regression. 

The examples use an options file 'config.cfg' that contain the configuration
for CURRENNT and a JSON file 'network.jsn' to specify the network topologies.
Make sure that you have built CURRENNT as described above before you run the
examples.

The following examples are provided:

+-----------------------------------------------------------------------------+
| 7.1 Multi-class classification / Speech recognition                         |
+-----------------------------------------------------------------------------+
The directory 'speech_recognition_chime' contains
a noisy small vocabulary speech recognition task with 51 words from the
2nd CHiME Speech Separation and Recognition Challenge. 

The two *.nc files contain the training and validation sequences as 39
Mel-frequency cepstral coefficient (MFCC) features per 10ms time step. Only the
data for speaker 1 is provided in order to keep the download size reasonable. 

The two directories 'no_subsampling' and 'subsampling' contain different
network topologies (with and without activation subsampling layers) that are
trained by executing the shell script 'run.sh' in each of these folders. 
On Windows, the batch file 'run.bat' does the same.

For a detailed description of the task and learning procedure, refer to

Martin Wöllmer, Felix Weninger, Jürgen Geiger, Björn Schuller, Gerhard Rigoll:
"Noise Robust ASR in Reverberated Multisource Environments Applying Convolutive
NMF and Long Short-Term Memory", Computer Speech and Language, Special Issue on
Speech Separation and Recognition in Multisource Environments, Elsevier, 17
pages, 2013.

+-----------------------------------------------------------------------------+
| 7.2 Regression / Autoencoding                                               |
+-----------------------------------------------------------------------------+
The directory 'speech_autoencoding_chime' contains an example for a regression
task, mapping noisy speech MFCC features to clean speech MFCC features, similar
to autoencoding.  The input data is the same as for the speech recognition
task, just the targets are different.  You can run the example via 'run.sh' 
or 'run.bat' as in the above.

For a detailed description of the task and learning procedure, refer to

Felix Weninger, Jürgen Geiger, Martin Wöllmer, Björn Schuller, Gerhard Rigoll:
"The Munich Feature Enhancement Approach to the 2013 CHiME Challenge Using
BLSTM Recurrent Neural Networks", Proc. 2nd CHiME Speech Separation and
Recognition Challenge held in conjunction with ICASSP 2013, IEEE, Vancouver,
Canada, 01.06.2013.
