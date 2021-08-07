from hyperparameters import (
    CategoricalHyperparam, ContinuousHyperparam,
    LogContinuousHyperparam, IntegerHyperparam
    )
from batch_jobs import (
    start_slurm_job, generate_working_dirname, copy_files_to_environment
    )
import yaml
import os
import getpass
import conf

tunables = []
shallow = False
num_nodes = 1
num_trials = 40

t_warn = CategoricalHyperparam(['data', 'T_warning'], [30,50,20.024,8,45,75,
                                                       10.024])
cut_ends = CategoricalHyperparam(['data', 'cut_shot_ends'], [False, True])
# for shallow
if shallow:
    num_nodes = 1
    shallow_model = CategoricalHyperparam(
        ['model', 'shallow_model', 'type'],
        ["svm", "random_forest", "xgboost", "mlp"])
    n_estimators = CategoricalHyperparam(
        ['model', 'shallow_model', 'n_estimators'],
        [5, 20, 50, 100, 300, 1000])
    max_depth = CategoricalHyperparam(
        ['model', 'shallow_model', 'max_depth'],
        [None, 3, 6, 10, 30, 100])
    C = LogContinuousHyperparam(['model', 'shallow_model', 'C'], 1e-3, 1e3)
    kernel = CategoricalHyperparam(['model', 'shallow_model', 'kernel'], [
                                   "rbf", "sigmoid", "linear", "poly"])
    xg_learning_rate = ContinuousHyperparam(
        ['model', 'shallow_model', 'learning_rate'], 0, 1)
    scale_pos_weight = CategoricalHyperparam(
        ['model', 'shallow_model', 'scale_pos_weight'], [1, 10.0, 100.0])
    num_samples = CategoricalHyperparam(
        ['model', 'shallow_model', 'num_samples'],
        [30000, 100000, 1000000, 2000000])
    hidden_size = CategoricalHyperparam(
        ['model', 'shallow_model', 'final_hidden_layer_size'], [5, 10, 20])
    hidden_num = CategoricalHyperparam(
        ['model', 'shallow_model', 'num_hidden_layers'], [2, 4])
    mlp_learning_rate = CategoricalHyperparam(
        ['model', 'shallow_model', 'learning_rate_mlp'],
        [0.001, 0.0001, 0.00001])
    mlp_regularization = CategoricalHyperparam(
        ['model', 'shallow_model', 'mlp_regularization'], [0.1, 0.003, 0.0001])
else:
    # for DL
    lr = LogContinuousHyperparam(['model', 'lr'], 1e-4, 1e-2)
    target = CategoricalHyperparam(['target'],['ttdinv','hinge','ttd'])
    patience = CategoricalHyperparam(['callbacks','patience'],[50,60,75,100,200,300])
    lr_decay_factor= CategoricalHyperparam(['model', 'lr_decay_factor'], [1.1,1.05,1.15,1.2,2,1.5])
    lr_decay_patience =CategoricalHyperparam(['model', 'lr_decay_patience'], [3,3,4,4,5,6,8,10])
    lr_decay = CategoricalHyperparam(['model', 'lr_decay'], [0.97, 0.985, 1.0])
    fac = CategoricalHyperparam(
        ['data', 'positive_example_penalty'], [0.0,1.0, 4.0, 16.0,1000000])
    batch_size = CategoricalHyperparam(['training', 'batch_size'], [8,8,8,8,16,32,64])
    predict_mode  = CategoricalHyperparam(['training', 'predict_mode'], ['derivative_target','smooth_target','derivative_target_norm','shift_target'])
    noise = CategoricalHyperparam(['training', 'noise'], [0.0,0.01,0.05,0.1,0.08])
    tcn_hidden = CategoricalHyperparam(['model', 'tcn_hidden'], [15,20,30,40,50,60])
    tcn_layers = CategoricalHyperparam(['model', 'tcn_layers'], [7,8,9,10,11,12,13])
    kernel_size_temporal = CategoricalHyperparam(['model', 'kernel_size_temporal'], [3,5,7,8,9,10,11,12,13])
    kernel_size_spatial = CategoricalHyperparam(['model', 'kernel_size_spatial'], [1,2,3,4,5])
 

    tt_lstm_hidden = CategoricalHyperparam(['model', 'tt_lstm_hidden'], [8,10,12,15,17,18,20,22,25,30,50,100,200])
    cell_order = CategoricalHyperparam(['model', 'cell_order'], [2,3,4,4,4,5,5,6])
    cell_steps = CategoricalHyperparam(['model', 'cell_steps'], [5,6,7,10,20,15,25,35,50,100,11,9])
    cell_rank = CategoricalHyperparam(['model', 'cell_rank'], [1,2,3,4,5,6,7,9,11])


 
    dropout_prob = CategoricalHyperparam(
        ['model', 'dropout_prob'], [0.01, 0.05,0.03,0.08,0.2, 0.1])
    conv_filters = CategoricalHyperparam(
        ['model', 'num_conv_filters'], [5,8,16,32,64, 128])
    conv_layers = CategoricalHyperparam(['model', 'num_conv_layers'], [1,2,3])
    rnn_layers = CategoricalHyperparam(['model', 'rnn_layers'], [1,1,2,3,4])
    rnn_size = CategoricalHyperparam(['model', 'rnn_size'], [16,32,48,50,64,72,80,96,128])
    dense_size = CategoricalHyperparam(['model', 'dense_size'], [64,128,200,160])
    dense_size_1d = CategoricalHyperparam(['model', 'dense_size_1d'], [16,40,32,64,128,200,160])
    dense_layers_1d = CategoricalHyperparam(['model', 'dense_layers_1d'], [1,2,2,3])
    simple_conv = CategoricalHyperparam(
        ['model', 'simple_conv'], [False, True])
    model_type = CategoricalHyperparam(
        ['model', 'model_type'], ['TCN'])
    extra_dense_input = CategoricalHyperparam(
        ['model', 'extra_dense_input'], [False, True])
    equalize_classes = CategoricalHyperparam(
        ['data', 'equalize_classes'], [False, True])
    t_min_warn = CategoricalHyperparam(['data', 'T_min_warn'],
                                       [30, 70, 200, 500, 1000])
    # rnn_length = CategoricalHyperparam(['model', 'length'], [32, 128])
    # tunables = [lr, lr_decay, fac, target, batch_size, dropout_prob]
    tunables = [t_warn,conv_filters,model_type,conv_layers,lr, lr_decay_factor,lr_decay_patience, fac, batch_size, equalize_classes,
                dropout_prob,patience]
    tunables += [simple_conv,rnn_layers,
                 rnn_size, dense_size,dense_size_1d,dense_layers_1d]
    tunables += [tcn_hidden,tcn_layers,
                 kernel_size_spatial,kernel_size_temporal]
    tunables += [tt_lstm_hidden,cell_steps,
                 cell_order,cell_rank]
    #tunables += [t_min_warn]
#tunables += [cut_ends, t_warn]

run_directory = "{}/{}/hyperparams/LM_tune/".format(
    conf.conf['fs_path'], getpass.getuser())
# "/home/{}/plasma-python/examples/".format(getpass.getuser())
template_path = os.environ['PWD']
conf_name = "conf.yaml"


def generate_conf_file(tunables, shallow, template_path="../", save_path="./",
                       conf_name="conf.yaml"):
    assert(template_path != save_path)
    with open(os.path.join(template_path, conf_name), 'r') as yaml_file:
        conf = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    for tunable in tunables:
        tunable.assign_to_conf(conf, save_path)
    # rely on early stopping to terminate training
    conf['training']['num_epochs'] = 1000
    # rely on early stopping to terminate training
    conf['training']['hyperparam_tuning'] = True
    conf['model']['shallow'] = shallow
    with open(os.path.join(save_path, conf_name), 'w') as outfile:
        yaml.dump(conf, outfile, default_flow_style=False)
    return conf


def get_executable_name_imposed_shallow(shallow):
    from conf import conf
    if shallow:
        executable_name = conf['paths']['shallow_executable']
        use_mpi = False
    else:
        executable_name = conf['paths']['executable']
        use_mpi = True
    return executable_name, use_mpi


working_directory = generate_working_dirname(run_directory)
os.makedirs(working_directory)

executable_name, _ = get_executable_name_imposed_shallow(shallow)
os.system(" ".join(["cp -p", os.path.join(template_path, conf_name),
                    working_directory]))
os.system(" ".join(["cp -p", os.path.join(template_path, executable_name),
                    working_directory]))
os.system(" ".join(["cp -p", os.path.join(template_path, '*.py'),
                    working_directory]))

os.chdir(working_directory)
print("Going into {}".format(working_directory))

for i in range(num_trials):
    subdir = working_directory + "/{}/".format(i)
    os.makedirs(subdir)
    copy_files_to_environment(subdir)
    print("Making modified conf")
    conf = generate_conf_file(tunables, shallow, working_directory, subdir,
                              conf_name)
    print("Starting job")
    os.system(" ".join(["cp -p", os.path.join(template_path, '*.py'),
                    subdir]))
    #start_slurm_job(subdir,num_nodes,i,conf,shallow,env_name=conf['env']['name'],env_type=conf['env']['type'],short=False)
    start_slurm_job(subdir,num_nodes,i,conf,shallow,cluster = 'tigergpu',env_name=conf['env']['name'],env_type=conf['env']['type'],short=False)
#    start_slurm_job(subdir, num_nodes, i, conf, shallow#,
                #   conf['env']['name'], conf['env']['type'],cluster='tigergpu')

print("submitted {} jobs.".format(num_trials))
