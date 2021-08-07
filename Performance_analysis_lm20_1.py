from __future__ import print_function
import matplotlib
#matplotlib.use('Agg')#for machines that don't have a display
from matplotlib import rc
#rc('font',**{'family':'serif','sans-serif':['Times']})
#rc('text', usetex=True)
import matplotlib.pyplot as plt
import os
from pprint import pprint
import numpy as np
from scipy import stats
from normalize import VarNormalizer as Normalizer 
from conf import conf
from shots import Shot,ShotList
from find_ntms import *
class PerformanceAnalyzer():
    def __init__(self,results_dir=None,shots_dir=None,i = 0,T_min_warn = None,T_max_warn = None, verbose = False,pred_ttd=False,conf=None):
        self.T_min_warn = T_min_warn
        self.T_max_warn = T_max_warn
        dt = conf['data']['dt']
        T_max_warn_def = int(round(conf['data']['T_warning']/dt))
        T_min_warn_def = conf['data']['T_min_warn']#int(round(conf['data']['T_min_warn']/dt))
        if T_min_warn == None:
            self.T_min_warn = T_min_warn_def
        if T_max_warn == None:
            self.T_max_warn = T_max_warn_def
        self.verbose = verbose
        self.results_dir = results_dir
        self.shots_dir = shots_dir
        self.i = i
        self.pred_ttd = pred_ttd
        self.saved_conf = conf
        self.conf = conf

        self.pred_train = None
        self.truth_train = None
        self.disruptive_train = None
        self.shot_list_train = None

        self.pred_test = None
        self.truth_test = None
        self.disruptive_test = None
        self.shot_list_test = None

        self.p_thresh_range = None

        self.normalizer = None



    def get_metrics_vs_p_thresh(self,mode):
        if mode == 'train':
            all_preds = self.pred_train
            all_truths = self.truth_train
            all_disruptive = self.disruptive_train


        elif mode == 'test':
            all_preds = self.pred_test
            all_truths = self.truth_test
            all_disruptive = self.disruptive_test

        return self.get_metrics_vs_p_thresh_custom(all_preds,all_truths,all_disruptive)



    def get_metrics_vs_p_thresh_custom(self,all_preds,all_truths,all_disruptive):
        return self.get_metrics_vs_p_thresh_fast(all_preds,all_truths,all_disruptive)
        P_thresh_range = self.get_p_thresh_range()
        correct_range = np.zeros_like(P_thresh_range)
        accuracy_range = np.zeros_like(P_thresh_range)
        fp_range = np.zeros_like(P_thresh_range)
        missed_range = np.zeros_like(P_thresh_range)
        early_alarm_range = np.zeros_like(P_thresh_range)
        
        for i,P_thresh in enumerate(P_thresh_range):
            correct,accuracy,fp_rate,missed,early_alarm_rate = self.summarize_shot_prediction_stats(P_thresh,all_preds,all_truths,all_disruptive)
            correct_range[i] = correct
            accuracy_range[i] = accuracy 
            fp_range[i] = fp_rate 
            missed_range[i] = missed
            early_alarm_range[i] = early_alarm_rate
        
        return correct_range,accuracy_range,fp_range,missed_range,early_alarm_range


    def get_p_thresh_range(self):
        #return self.conf['data']['target'].threshold_range(self.conf['data']['T_warning'])
        if np.any(self.p_thresh_range) == None:
                all_preds_tr = self.pred_train
                all_truths_tr = self.truth_train
                all_disruptive_tr = self.disruptive_train
                all_preds_te = self.pred_test
                all_truths_te = self.truth_test
                all_disruptive_te = self.disruptive_test
        
                early_th_tr,correct_th_tr,late_th_tr,nd_th_tr = self.get_threshold_arrays(all_preds_tr,all_truths_tr,all_disruptive_tr)
                early_th_te,correct_th_te,late_th_te,nd_th_te = self.get_threshold_arrays(all_preds_te,all_truths_te,all_disruptive_te)
                all_thresholds = np.sort(np.concatenate((early_th_tr,correct_th_tr,late_th_tr,nd_th_tr,early_th_te,correct_th_te,late_th_te,nd_th_te)))
                self.p_thresh_range = all_thresholds
        #print(np.unique(self.p_thresh_range))
        return self.p_thresh_range
                

    def get_metrics_vs_p_thresh_fast(self,all_preds,all_truths,all_disruptive):
        all_disruptive = np.array(all_disruptive)        
        if self.pred_train is not None:
                p_thresh_range = self.get_p_thresh_range()
        else:
                early_th,correct_th,late_th,nd_th = self.get_threshold_arrays(all_preds,all_truths,all_disruptive)
                p_thresh_range = np.sort(np.concatenate((early_th,correct_th,late_th,nd_th)))
        correct_range = np.zeros_like(p_thresh_range)
        accuracy_range = np.zeros_like(p_thresh_range)
        fp_range = np.zeros_like(p_thresh_range)
        missed_range = np.zeros_like(p_thresh_range)
        early_alarm_range = np.zeros_like(p_thresh_range)

        early_th,correct_th,late_th,nd_th = self.get_threshold_arrays(all_preds,all_truths,all_disruptive)

        for i,thresh in enumerate(p_thresh_range):
            #correct,accuracy,fp_rate,missed,early_alarm_rate = self.summarize_shot_prediction_stats(thresh,all_preds,all_truths,all_disruptive)
            correct,accuracy,fp_rate,missed,early_alarm_rate = self.get_shot_prediction_stats_from_threshold_arrays(early_th,correct_th,late_th,nd_th,thresh)
            correct_range[i] = correct
            accuracy_range[i] = accuracy 
            fp_range[i] = fp_rate 
            missed_range[i] = missed
            early_alarm_range[i] = early_alarm_rate
        
        return correct_range,accuracy_range,fp_range,missed_range,early_alarm_range

    def get_shot_prediction_stats_from_threshold_arrays(self,early_th,correct_th,late_th,nd_th,thresh):
        indices = np.where(np.logical_and(correct_th > thresh,early_th <= thresh))[0]
        FPs = np.sum(nd_th > thresh)
        TNs = len(nd_th) - FPs 

        earlies = np.sum(early_th > thresh) 
        TPs = np.sum(np.logical_and(early_th <= thresh,correct_th > thresh)) 
        lates = np.sum(np.logical_and(np.logical_and(early_th <= thresh,correct_th <= thresh),late_th > thresh))
        FNs = np.sum(np.logical_and(np.logical_and(early_th <= thresh,correct_th <= thresh),late_th <= thresh))

        return self.get_accuracy_and_fp_rate_from_stats(TPs,FPs,FNs,TNs,earlies,lates)


    def get_shot_difficulty(self,preds,truths,disruptives):
        disruptives = np.array(disruptives)
        d_early_thresholds, d_correct_thresholds,d_late_thresholds, nd_thresholds = self.get_threshold_arrays(preds,truths,disruptives)
        d_thresholds = np.maximum(d_early_thresholds,d_correct_thresholds)
        #rank shots by difficulty. rank 1 is assigned to lowest value, should be highest difficulty
        d_ranks = stats.rankdata(d_thresholds,method='min')#difficulty is highest when threshold is low, can't detect disruption
        nd_ranks = stats.rankdata(-nd_thresholds,method='min')#difficulty is highest when threshold is high, can't avoid false positive 
        ranking_fac = self.saved_conf['training']['ranking_difficulty_fac']
        facs_d = np.linspace(ranking_fac,1,len(d_ranks))[d_ranks-1]
        facs_nd = np.linspace(ranking_fac,1,len(nd_ranks))[nd_ranks-1]
        ret_facs = np.ones(len(disruptives))
        ret_facs[disruptives] = facs_d
        ret_facs[~disruptives] = facs_nd
        #print("setting shot difficulty")
        #print(disruptives)
        #print(d_thresholds)
        #print(nd_thresholds)
        #print(ret_facs)
        return ret_facs

    def get_threshold_arrays(self,preds,truths,disruptives):
        num_d = np.sum(disruptives)
        num_nd = np.sum(~disruptives)
        nd_thresholds = [] 
        d_early_thresholds = [] 
        d_correct_thresholds = [] 
        d_late_thresholds = [] 
        for i in range(len(preds)):
            pred = 1.0*preds[i]
            truth = truths[i]
            pred[:self.get_ignore_indices()] = -np.inf
            is_disruptive = disruptives[i] 
            if is_disruptive:
                max_acceptable = self.create_acceptable_region(truth,'max')
                min_acceptable = self.create_acceptable_region(truth,'min')
                correct_indices = np.logical_and(max_acceptable, ~min_acceptable)
                early_indices = ~max_acceptable
                late_indices = min_acceptable
                if np.sum(late_indices) == 0:
                        d_late_thresholds.append(-np.inf)
                else:
                        d_late_thresholds.append(np.max(pred[late_indices]))
                if np.sum(early_indices) == 0:
                        d_early_thresholds.append(-np.inf)
                else:
                        d_early_thresholds.append(np.max(pred[early_indices]))
                        
                d_correct_thresholds.append(np.max(pred[correct_indices]))
            else:
                nd_thresholds.append(np.max(pred))
        return np.array(d_early_thresholds), np.array(d_correct_thresholds),np.array(d_late_thresholds), np.array(nd_thresholds)
     





    def summarize_shot_prediction_stats_by_mode(self,P_thresh,mode,verbose=False):

        if mode == 'train':
            all_preds = self.pred_train
            all_truths = self.truth_train
            all_disruptive = self.disruptive_train


        elif mode == 'test':
            all_preds = self.pred_test
            all_truths = self.truth_test
            all_disruptive = self.disruptive_test

        return self.summarize_shot_prediction_stats(P_thresh,all_preds,all_truths,all_disruptive,verbose)


    def summarize_shot_prediction_stats(self,P_thresh,all_preds,all_truths,all_disruptive,verbose=False):
        TPs,FPs,FNs,TNs,earlies,lates = (0,0,0,0,0,0)

        for i in range(len(all_preds)):
            preds = all_preds[i]
            truth = all_truths[i]
            is_disruptive = all_disruptive[i]

            
            TP,FP,FN,TN,early,late = self.get_shot_prediction_stats(P_thresh,preds,truth,is_disruptive)
            TPs += TP
            FPs += FP
            FNs += FN
            TNs += TN
            earlies += early
            lates += late
            
        disr = earlies + lates + TPs + FNs
        nondisr = FPs + TNs
        if verbose:
            print('total: {}, tp: {} fp: {} fn: {} tn: {} early: {} late: {} disr: {} nondisr: {}'.format(len(all_preds),TPs,FPs,FNs,TNs,earlies,lates,disr,nondisr))
       
        return self.get_accuracy_and_fp_rate_from_stats(TPs,FPs,FNs,TNs,earlies,lates,verbose)



    #we are interested in the predictions of the *first alarm*
    def get_shot_prediction_stats(self,P_thresh,pred,truth,is_disruptive):
        if self.pred_ttd:
            predictions = pred < P_thresh
        else:
            predictions = pred > P_thresh
        print(predictions.shape)
        predictions = np.reshape(predictions,(len(predictions)))
        
        max_acceptable = self.create_acceptable_region(truth,'max')
        min_acceptable = self.create_acceptable_region(truth,'min')
        
        early = late = TP = TN = FN = FP = 0
      
        positives = self.get_positives(predictions)#where(predictions)[0]
        if len(positives) == 0:
            if is_disruptive:
                FN = 1
            else:
                TN = 1
        else:
            if is_disruptive:
                first_pred_idx = positives[0]
                print(first_pred_idx)
                print(max_acceptable)
                if max_acceptable[first_pred_idx] and ~min_acceptable[first_pred_idx]:
                    TP = 1
                elif min_acceptable[first_pred_idx]:
                    late = 1
                elif ~max_acceptable[first_pred_idx]:
                    early = 1
            else:
                FP = 1
        return TP,FP,FN,TN,early,late

    def get_ignore_indices(self):
        return self.saved_conf['model']['ignore_timesteps']


    def get_positives(self,predictions):
        indices = np.arange(len(predictions))
        return np.where(np.logical_and(predictions,indices >= self.get_ignore_indices()))[0]


    def create_acceptable_region(self,truth,mode):
        if mode == 'min':
            acceptable_timesteps = self.T_min_warn
        elif mode == 'max':
            acceptable_timesteps = self.T_max_warn
        else:
            print('Error Invalid Mode for acceptable region')
            exit(1) 

        acceptable = np.zeros_like(truth,dtype=bool)
        if acceptable_timesteps > 0:
            acceptable[-acceptable_timesteps:] = True
        return acceptable


    def get_accuracy_and_fp_rate_from_stats(self,tp,fp,fn,tn,early,late,verbose=False):
        total = tp + fp + fn + tn + early + late
        disr = early + late + tp + fn 
        nondisr = fp + tn
        
        if disr == 0:
            early_alarm_rate = 0
            missed = 0
            accuracy = 0 
        else:
            early_alarm_rate = 1.0*early/disr
            missed = 1.0*(late + fn)/disr
            accuracy = 1.0*tp/disr
        if nondisr == 0:
            fp_rate = 0
        else: 
            fp_rate = 1.0*fp/nondisr
        correct = 1.0*(tp + tn)/total
        
        if verbose:
            print('accuracy: {}'.format(accuracy))
            print('missed: {}'.format(missed))
            print('early alarms: {}'.format(early_alarm_rate))
            print('false positive rate: {}'.format(fp_rate))
            print('correct: {}'.format(correct))

        return correct,accuracy,fp_rate,missed,early_alarm_rate
    def find_thresh_range(self,yprime = None,mode = None,resolution = 50,pred_index = 0):
        mini = 10000
        maxi = -1000
        if mode ==None:
            ys = yprime
        elif mode in [ 'test','Test']:
            ys = self.pred_test
        elif mode in ['validation','Validation']:
            ys = self.pred_validation

        for p in ys:
            mini = min(mini,np.amin(p[:,pred_index]))
            maxi = max(maxi,np.amax(p[:,pred_index]))

        return np.arange(resolution)*(maxi-mini)/resolution+mini

    def find_truth_positive_time(self,mode = None,pred_index=0,ygold = None,truth_positive_threshold=10):
        positive_start_time = []
        if mode ==None:
            ys = ygold
        positive = 0
        negative = 0
        for y in ys:
            yi = np.array(y[:,pred_index])
           # print(yi.shape)
            arr = np.argwhere(yi>truth_positive_threshold)
           # print('arrshape',arr.shape)
            if arr.shape[0] == 0:
                positive_start_time.append(-1)
                negative+=1
            else:
                mode_time  = int(arr[0])
            #    print('!!!!!!!!!!!!!!!!!!!!!FOUND mode time',mode_time)
                positive_start_time.append(mode_time)
                positive +=1
        return np.array(positive_start_time),positive,negative


    def calculate_sig_roc(self,yprime = None, ygold=None,mode =None,pred_index =1,truth_positive_threshold = 8):
       # early_th_tr,correct_th_tr,late_th_tr,nd_th_tr = self.get_threshold_arrays(all_preds_tr,all_truths_tr,tearing_tr)
        preds_te = [p[:,pred_index] for p in yprime]
        truth_te = [p[:,pred_index] for p in ygold]
        tearing_te = [np.amax(p[:,pred_index])>truth_positive_threshold for p in ygold]
        tearing_te = (np.array(tearing_te))
        #print(tearing_te)
        early_th_te,correct_th_te,late_th_te,nd_th_te = self.get_threshold_arrays(preds_te,truth_te,tearing_te)
        thresh_range = np.sort(np.concatenate((early_th_te,correct_th_te,late_th_te,nd_th_te)))
        #print('Finished finding threshold range.....')
        truth_positive_time,p_truth,n_truth = self.find_truth_positive_time(mode = mode,pred_index = pred_index,ygold = ygold,truth_positive_threshold = truth_positive_threshold)
        TPR = []
        FPR = []
        TPR_relaxed = []
        #print('...............p_truth,n_truth')
        #print(p_truth,n_truth)
        thresh_range = thresh_range[::-1]
        thresh_range = thresh_range[np.where(thresh_range>0)]
        #print(thresh_range)
        for t in thresh_range:
         #   print('t:',t)
            pred_positive_time,p_pred,n_pred = self.find_truth_positive_time(mode = mode,pred_index = pred_index,ygold = yprime,truth_positive_threshold = t)
            TP = len(np.argwhere((truth_positive_time>=0)& (pred_positive_time>=0)&(pred_positive_time<=truth_positive_time)))
            TP_relaxed = len(np.argwhere((truth_positive_time>=0)& (pred_positive_time>=0)))
            FP = len(np.argwhere((truth_positive_time<0)& (pred_positive_time>=0)))
         #   print('len(truth_positive_time)',len(truth_positive_time))
         #   print('len(pred_positive_time)',len(pred_positive_time))
         #   print('truth_positive_time',truth_positive_time)
         #   print('pred_positive_time',pred_positive_time)
         #   print('p_pred,n_pred')
         #   print(p_pred,n_pred)
         #   print('TP,FP')
         #   print(TP,FP)
            TPR.append(TP/p_truth)
            FPR.append(FP/n_truth)
            TPR_relaxed.append(TP_relaxed/p_truth)
        auroc =  np.trapz(TPR,x=FPR)
        auroc_relaxed = np.trapz(TPR_relaxed,x = FPR)
        return thresh_range, TPR, FPR,auroc,TPR_relaxed,auroc_relaxed
    def check_modes(self,mode = 'test',bar_plot = True):
        print('Checking target modes...........................................')
        print(self.pred_test.shape)
        normalization_factor = 1.22308913
        normalization_factor_pred = 1.22308913
        pred_norm_factor = 2.5
        max_mode = []
        total_ntm  = 0
        found_mode = []
        max_array = [] 
        max_mode_p = []
        total_ntm_p  = 0
        found_mode_p = []
        max_array_p = [] 
        FP=0 
        TP=0
        TN = 0
        FN =0
        late =0
        alarm_times = []
        shot_list = self.shot_list_test
        for i,pred in enumerate(self.truth_test):
            print(shot_list[i].number,'====TRUTH=====================================================')
            print(pred[:,0].shape)
            max_n1 = (max(pred[:,0])*normalization_factor)
            max_array.append(max_n1)
            if max_n1>10:
                max_mode.append(pred[:,0]) 
            ntm_events = find_ntm_events(np.arange(pred.shape[0]),pred[:,0],threshold = 10/normalization_factor)
           
            print(len(ntm_events))
            total_ntm += len( ntm_events)
            if len(ntm_events)>0:
                found_mode.append(ntm_events)
            print('=================PREDICTIONs')
            pred = self.pred_test[i]*pred_norm_factor
            print(pred[:,0].shape)
            max_n1 = (max(pred[:,0])*normalization_factor_pred)
            max_array_p.append(max_n1)
            if max_n1>10:
                max_mode_p.append(pred[:,0]) 
            ntm_events_p = find_ntm_events(np.arange(pred.shape[0]),pred[:,0],threshold = 10/normalization_factor_pred)
            shot_type =''
            if len(ntm_events)==0 and len(ntm_events_p)>0:
                FP+=1
                shot_type = 'FP'
            elif len(ntm_events) ==0 and len(ntm_events_p)==0:
                TN+=1
                shot_type = 'TN'
            elif len(ntm_events)>0 and len(ntm_events_p)==0:
                shot_type = 'FN'
                FN+=1
            elif ntm_events_p[0]['begin']<ntm_events[0]['begin']:
                TP+=1
                shot_type = 'TP'
                alarm_time = ntm_events_p[0]['begin']-ntm_events[0]['begin']
                alarm_times.append(alarm_time)
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            else:
                late +=1
                shot_type = 'late'
            print(shot_type) 
            print(len(ntm_events_p))
            total_ntm += len( ntm_events_p)
            if len(ntm_events_p)>0:
                found_mode_p.append(ntm_events)

        print('mean of truth max',np.mean(max_array))
        print('median of truth max',np.median(max_array))
        print('mean of predicted max',np.mean(max_array_p))
        print('median of predicted max',np.median(max_array_p))
        print('Total mode max detected (1 every shot):',len(max_mode))
        print('Total mode detected (multiple every shot):',total_ntm,len(found_mode))
        print('Total mode max detected PREDICT(1 every shot):',len(max_mode_p))
        print('Total mode detected PREDICT (multiple every shot):',total_ntm,len(found_mode_p))

        print('Finished checking target modes..................................')
        print('TP',TP)
        print('FP',FP)
        print('TN',TN)
        print('FN',FN)
        print('late',late)
        print('TPR',(TP+late)/(late+TP+FN))
        print('FPR',FP/(TN+FP))
        print('acc',(TP+TN+late)/(TP+FP+TN+FN+late))
        print('median alarm time:',np.median(alarm_times))
        print('mean alarm time:',np.mean(alarm_times))
        if bar_plot:
            hist,bins = np.histogram(alarm_times,bins = 30,range = (-800,0))
            print (alarm_times)
            print(bins)
            print(hist)
            plt.bar(bins[:-1],hist,edgecolor = 'black',facecolor='green',width = 800/30.)

            plt.savefig('barh.png')
        return 
    def load_ith_file(self):
        results_files = os.listdir(self.results_dir)
        print(results_files)
        print("Loading results file {}".format(self.results_dir + results_files[self.i]))
        dat = np.load(self.results_dir + results_files[self.i],
                      allow_pickle=True)

        if self.verbose:
            print('configuration: {} '.format(dat['conf']))

        self.pred_train = dat['y_prime_train']
        self.truth_train = dat['y_gold_train']
        self.disruptive_train = dat['disruptive_train']
        self.pred_test = dat['y_prime_test']
        self.truth_test = dat['y_gold_test']
        self.disruptive_test = dat['disruptive_test']
        self.shot_list_test = ShotList(dat['shot_list_test'][()])
        self.shot_list_train = ShotList(dat['shot_list_train'][()])
        self.saved_conf = dat['conf'][()]
        self.conf['data']['T_warning'] = self.saved_conf['data']['T_warning'] #all files must agree on T_warning due to output of truth vs. normalized shot ttd.
        for mode in ['test','train']:
            print('{}: loaded {} shot ({}) disruptive'.format(mode,self.get_num_shots(mode),self.get_num_disruptive_shots(mode)))
        if self.verbose:
            self.print_conf()
        #self.assert_same_lists(self.shot_list_test,self.truth_test,self.disruptive_test)
        #self.assert_same_lists(self.shot_list_train,self.truth_train,self.disruptive_train)

    def assert_same_lists(self,shot_list,truth_arr,disr_arr):
        assert(len(shot_list) == len(truth_arr))
        for i in range(len(shot_list)):
                shot_list.shots[i].restore("/tigress/jk7/processed_shots/")
                s = shot_list.shots[i].ttd
                if not truth_arr[i].shape[0] == s.shape[0]-30:
                        print(i)
                        print(shot_list.shots[i].number)
                        print((s.shape,truth_arr[i].shape,disr_arr[i]))
                assert(truth_arr[i].shape[0] == s.shape[0]-30)
        print("Same Shape!")
   
    def print_conf(self):
        pprint(self.saved_conf) 

    def get_num_shots(self,mode):
        if mode == 'test':
            return len(self.disruptive_test)
        if mode == 'train':
            return len(self.disruptive_train)

    def get_num_disruptive_shots(self,mode):
        if mode == 'test':
            return sum(self.disruptive_test)
        if mode == 'train':
            return sum(self.disruptive_train)


    def hist_alarms(self,alarms,title_str='alarms',save_figure=False,linestyle='-'):
        fontsize=15
        T_min_warn = self.T_min_warn
        T_max_warn = self.T_max_warn
        if len(alarms) > 0:
            alarms = alarms / 1000.0
            alarms = np.sort(alarms)
            T_min_warn /= 1000.0
            T_max_warn /= 1000.0
            plt.figure()
            alarms += 0.0001
            bins=np.logspace(np.log10(min(alarms)),np.log10(max(alarms)),40)
            #bins=linspace(min(alarms),max(alarms),100)
            #        hist(alarms,bins=bins,alpha=1.0,histtype='step',normed=True,log=False,cumulative=-1)
            #
            plt.step(np.concatenate((alarms[::-1], alarms[[0]])), 1.0*np.arange(alarms.size+1)/(alarms.size),linestyle=linestyle,linewidth=1.5)

            plt.gca().set_xscale('log')
          #  plt.axvline(T_min_warn,color='r',linewidth=0.5)
            #if T_max_warn < np.max(alarms):
            #    plt.axvline(T_max_warn,color='r',linewidth=0.5)
            plt.xlabel('Time to disruption [s]',size=fontsize)
            plt.ylabel('Fraction of detected disruptions',size=fontsize)
            plt.xlim([1e-4,4e1])#max(alarms)*10])
            plt.ylim([0,1])
            plt.grid()
            plt.title(title_str)
            plt.setp(plt.gca().get_yticklabels(),fontsize=fontsize)
            plt.setp(plt.gca().get_xticklabels(),fontsize=fontsize)
            plt.show()
            if save_figure:
                plt.savefig('accum_disruptions.png',dpi=200,bbox_inches='tight')
        else:
            print(title_str + ": No alarms!")



    def gather_first_alarms(self,P_thresh,mode):
        if mode == 'train':
            pred_list = self.pred_train 
            disruptive_list = self.disruptive_train 
        elif mode == 'test':
            pred_list = self.pred_test 
            disruptive_list = self.disruptive_test 
        
        alarms = []
        disr_alarms = []
        nondisr_alarms = []
        for i in range(len(pred_list)):
            pred = pred_list[i]
            if self.pred_ttd:
                predictions = pred < P_thresh
            else:
                predictions = pred > P_thresh
            print(predictions.shape)
            predictions = np.reshape(predictions,(len(predictions),))
            positives = self.get_positives(predictions) #where(predictions)[0]
            if len(positives) > 0:
                alarm_ttd = len(pred) - 1.0 - positives[0]
                alarms.append(alarm_ttd)
                if disruptive_list[i]:
                    disr_alarms.append(alarm_ttd)
                else:
                    nondisr_alarms.append(alarm_ttd)
            else:
                if disruptive_list[i]:
                    disr_alarms.append(-1)
        return np.array(alarms),np.array(disr_alarms),np.array(nondisr_alarms)
                

    def compute_tradeoffs_and_print(self,mode):
        P_thresh_range = self.get_p_thresh_range()
        correct_range, accuracy_range, fp_range,missed_range,early_alarm_range = self.get_metrics_vs_p_thresh(mode)
        fp_threshs = [0.01,0.05,0.1]
        missed_threshs = [0.01,0.05,0.0]
 #       missed_threshs = [0.01,0.05,0.1,0.2,0.3]

        #first index where...
        for fp_thresh in fp_threshs: 
            print('============= FP RATE < {} ============='.format(fp_thresh))
            if(any(fp_range < fp_thresh)):
                idx = np.where(fp_range <= fp_thresh)[0][0]
                P_thresh_opt = P_thresh_range[idx]
                self.summarize_shot_prediction_stats_by_mode(P_thresh_opt,mode,verbose=True)
                print('============= AT P_THRESH = {} ============='.format(P_thresh_opt))
            else:
                print('No such P_thresh found') 
            print('')

        #last index where
        for missed_thresh in missed_threshs: 
            print('============= MISSED RATE < {} ============='.format(missed_thresh))
            if(any(missed_range < missed_thresh)):
                idx = np.where(missed_range <= missed_thresh)[0][-1]
                P_thresh_opt = P_thresh_range[idx]
                self.summarize_shot_prediction_stats_by_mode(P_thresh_opt,mode,verbose=True)
                print('============= AT P_THRESH = {} ============='.format(P_thresh_opt))
            else:
                print('No such P_thresh found')
            print('')

        print('============== Crossing Point: ==============')
        print('============= TEST PERFORMANCE: =============')
        idx = np.where(missed_range <= fp_range)[0][-1]
        P_thresh_opt = P_thresh_range[idx]
        self.summarize_shot_prediction_stats_by_mode(P_thresh_opt,mode,verbose=True)
        P_thresh_ret = P_thresh_opt
        return P_thresh_ret


    def compute_tradeoffs_and_print_from_training(self):
        P_thresh_range = self.get_p_thresh_range()
        correct_range, accuracy_range, fp_range,missed_range,early_alarm_range = self.get_metrics_vs_p_thresh('train')

        fp_threshs = [0.01,0.05,0.1]
        missed_threshs = [0.01,0.05,0.0]
#        missed_threshs = [0.01,0.05,0.1,0.2,0.3]
        P_thresh_default = 0.03
        P_thresh_ret = P_thresh_default

        first_idx = 0 if not self.pred_ttd else -1
        last_idx = -1 if not self.pred_ttd else 0

        #first index where...
        for fp_thresh in fp_threshs: 

            print('============= TRAINING FP RATE < {} ============='.format(fp_thresh))
            print('============= TEST PERFORMANCE: =============')
            if(any(fp_range < fp_thresh)):
                idx = np.where(fp_range <= fp_thresh)[0][first_idx]
                P_thresh_opt = P_thresh_range[idx]
                self.summarize_shot_prediction_stats_by_mode(P_thresh_opt,'test',verbose=True)
                print('============= AT P_THRESH = {} ============='.format(P_thresh_opt))
            else:
                print('No such P_thresh found')
            P_thresh_opt = P_thresh_default
            print('')

        #last index where
        for missed_thresh in missed_threshs: 

            print('============= TRAINING MISSED RATE < {} ============='.format(missed_thresh))
            print('============= TEST PERFORMANCE: =============')
            if(any(missed_range < missed_thresh)):
                idx = np.where(missed_range <= missed_thresh)[0][last_idx]
                P_thresh_opt = P_thresh_range[idx]
                self.summarize_shot_prediction_stats_by_mode(P_thresh_opt,'test',verbose=True)
                if missed_thresh == 0.05:
                    P_thresh_ret = P_thresh_opt
                print('============= AT P_THRESH = {} ============='.format(P_thresh_opt))
            else:
                print('No such P_thresh found')
            P_thresh_opt = P_thresh_default
            print('')

        print('============== Crossing Point: ==============')
        print('============= TEST PERFORMANCE: =============')
        if(any(missed_range <= fp_range)):
            idx = np.where(missed_range <= fp_range)[0][last_idx]
            P_thresh_opt = P_thresh_range[idx]
            self.summarize_shot_prediction_stats_by_mode(P_thresh_opt,'test',verbose=True)
            P_thresh_ret = P_thresh_opt
            print('============= AT P_THRESH = {} ============='.format(P_thresh_opt))
        else:
            print('No such P_thresh found')
        return P_thresh_ret


    def compute_tradeoffs_and_plot(self,mode,save_figure=True,plot_string='',linestyle="-",thres_opt=None):
        correct_range, accuracy_range, fp_range,missed_range,early_alarm_range = self.get_metrics_vs_p_thresh(mode)

        return self.tradeoff_plot(accuracy_range,missed_range,fp_range,early_alarm_range,save_figure=save_figure,plot_string=plot_string,linestyle=linestyle,thres_opt=thres_opt)

    def get_prediction_type(self,TP,FP,FN,TN,early,late):
        if TP:
            return 'TP'
        elif FP:
            return 'FP'
        elif FN:
            return 'FN'
        elif TN:
            return 'TN'
        elif early:
            return 'early'
        elif late:
            return 'late'

    def plot_individual_shot(self,P_thresh_opt,shot_num,normalize=True,plot_signals=True,tmin = None):
        success = False
        for mode in ['test','train']:
            if mode == 'test':
                pred = self.pred_test
                truth = self.truth_test
                is_disruptive = self.disruptive_test
                shot_list = self.shot_list_test
            else:
                pred = self.pred_train
                truth = self.truth_train
                is_disruptive = self.disruptive_train
                shot_list = self.shot_list_train
            for i,shot in enumerate(shot_list):
                if shot.number == shot_num:
                    t = truth[i]
                    p = pred[i]
                    is_disr = is_disruptive[i]

                    TP,FP,FN,TN,early,late =self.get_shot_prediction_stats(P_thresh_opt,p[:0],t,is_disr)
                    prediction_type = self.get_prediction_type(TP,FP,FN,TN,early,late)
                    print(prediction_type)
                    self.plot_shot(shot,True,normalize,t,p,P_thresh_opt,prediction_type,extra_filename = '_indiv',tmin = tmin)
                    success = True
        if not success:
            print("Shot {} not found".format(shot_num))
                

    def get_prediction_type_for_individual_shot(self,P_thresh,shot,mode='test'):
        p,t,is_disr = self.get_pred_truth_disr_by_shot(shot)

        TP,FP,FN,TN,early,late =self.get_shot_prediction_stats(P_thresh,p[:,0],t,is_disr)
        prediction_type = self.get_prediction_type(TP,FP,FN,TN,early,late)
        return prediction_type


    def example_plots(self,P_thresh_opt,mode='test',types_to_plot = ['FP'],max_plot = 5,normalize=True,plot_signals=True,extra_filename=''):
        if mode == 'test':
            pred = self.pred_test
            truth = self.truth_test
            is_disruptive = self.disruptive_test
            shot_list = self.shot_list_test
        else:
            pred = self.pred_train
            truth = self.truth_train
            is_disruptive = self.disruptive_train
            shot_list = self.shot_list_train
        plotted = 0
        iterate_arr = np.arange(len(truth))
        np.random.shuffle(iterate_arr)
        for i in iterate_arr:
            t = truth[i]
            p = pred[i]
            is_disr = is_disruptive[i]
            shot = shot_list.shots[i]
            print('printing_out shot number',shot.number)
            print(t.shape)
            TP,FP,FN,TN,early,late =self.get_shot_prediction_stats(P_thresh_opt,p[:,0],t[:,0],is_disr)
            prediction_type = self.get_prediction_type(TP,FP,FN,TN,early,late)
            if not all(_ in set(['FP','TP','FN','TN','late','early','any']) for _ in types_to_plot):
                print('warning, unkown types_to_plot')
                return
            if ('any' in types_to_plot or prediction_type in types_to_plot) and plotted < max_plot:
                if plot_signals:
                    self.plot_shot(shot,save_fig=True,normalize=normalize,truth=t,prediction=p,P_thresh_opt=P_thresh_opt,prediction_type=prediction_type,extra_filename=extra_filename)
                else:
                    plt.figure()
                    plt.semilogy((t+0.001)[::-1],label='ground truth')
                    plt.plot(p[::-1],'g',label='neural net prediction')
                    plt.axvline(self.T_min_warn,color='r',label='max warning time')
                    plt.axvline(self.T_max_warn,color='r',label='min warning time')
                    plt.axhline(P_thresh_opt,color='k',label='trigger threshold')
                    plt.xlabel('TTD [ms]')
                    plt.legend(loc = (1.0,0.6))
                    plt.ylim([1e-7,1.1e0])
                    plt.grid()
                    plt.savefig('fig_{}.png'.format(shot.number),bbox_inches='tight')
                plotted += 1

    def plot_shot(self,shot,save_fig=True,normalize=True,truth=None,prediction=None,P_thresh_opt=None,prediction_type='',extra_filename='',tmin = None):
        print('plotting shot,',shot,prediction_type,prediction.shape)
        if self.normalizer is None and normalize:
            if self.conf is not None:
                self.saved_conf['paths']['normalizer_path'] = self.conf['paths']['normalizer_path']
            nn = Normalizer(self.saved_conf)
            nn.train()
            self.normalizer = nn
            self.normalizer.set_inference_mode(True)
        if tmin == None:
            tmin = 1000
        if(shot.previously_saved(self.shots_dir)):
            shot.restore(self.shots_dir)
            if shot.signals_dict is not None: #make sure shot was saved with data
                t_disrupt = shot.t_disrupt
                is_disruptive =  shot.is_disruptive
                if normalize:
                    self.normalizer.apply(shot)
    
                use_signals = self.saved_conf['paths']['use_signals']
                all_signals = self.saved_conf['paths']['all_signals']
                fontsize= 18
                lower_lim = tmin #len(pred)
                plt.close()
                colors = ['b','green','red','c','m','orange','k','y']
                lss = ["-","--"]
                #f,axarr = plt.subplots(len(use_signals)+1,1,sharex=True,figsize=(10,15))#, squeeze=False)
                f,axarr = plt.subplots(4+1,1,sharex=True,figsize=(18,18))#,squeeze=False)#, squeeze=False)
                #plt.title(prediction_type)
                #assert(np.all(shot.ttd.flatten() == truth.flatten()))
                xx = range((prediction.shape[0]))
                j=0 #list(reversed(range(len(pred))))
                j1=0
                p0=0
                for i,sig_target in enumerate(all_signals):
                    if sig_target.description== 'n1 finite frequency signals_10ms': 
#'Locked mode amplitude':
                       target_plot=shot.signals_dict[sig_target]##[:,0]
                       target_plot=target_plot[:,0]
                       print(target_plot.shape)
                    elif sig_target.description== 'Locked mode amplitude':
                       lm_plot=shot.signals_dict[sig_target]##[:,0]
                       lm_plot=lm_plot[:,0]
                plot_des=['plasma current','poloidal beta from EFIT','Minimum safety factor','q95 safety factor']
                plot_des+=['Plasma density','Radiated Power Edge','Radiated Power Core','Input Power (beam for d3d)']
                plot_des+=['Electron temperature profile','Electron density profile']
                plot_signals=[]
                for sig in use_signals:
                     if sig.description in plot_des:
                         plot_signals.append(sig)
                for i,sig in enumerate(plot_signals):
                    num_channels = sig.num_channels
                    sig_arr = shot.signals_dict[sig]
                    legend=[]
                    if num_channels == 1:
                        j=i//4
                        ax = axarr[j]
        #                 if j == 0:
                        ax.plot(xx,sig_arr[:,0],linewidth=2,color=colors[i%4],label=sig.description)#,linestyle=lss[j],color=colors[j])
        #                 else:
        #                     ax.plot(xx,sig_arr[:,0],linewidth=2)#,linestyle=lss[j],color=colors[j],label = labels[sig])
                        #if np.min(sig_arr[:,0]) < -100000:
                        if j==0:
                          ax.set_ylim([-1,17])
                          ax.set_yticks([0,5,10,15])
                        else:
                          ax.set_ylim([-1,8])
                          ax.set_yticks([0,3,6])
        #                 ax.set_ylabel(labels[sig],size=fontsize)
                        ax.legend()
                    else:
                        j=-2-j1
                        j1+=1
                        ax = axarr[j]
                        ax.imshow(sig_arr[:,:].T, aspect='auto', label = sig.description,cmap="inferno" )
                        ax.set_ylim([0,num_channels])
                        ax.text(lower_lim+200, 45, sig.description, bbox={'facecolor': 'white', 'pad': 10},fontsize=fontsize)
                        ax.set_yticks([0,num_channels/2])
                        ax.set_yticklabels(["0","0.5"])
                        ax.set_ylabel("$\\rho$",size=fontsize)
                    ax.legend(loc="center left",labelspacing=0.1,bbox_to_anchor=(1,0.5),fontsize=fontsize,frameon=False)
                   # ax.axvline(len(truth)-self.T_min_warn,color='r')
                   # ax.axvline(p0,linestyle='--',color='darkgrey')
                    plt.setp(ax.get_xticklabels(),visible=False)
                    plt.setp(ax.get_yticklabels(),fontsize=fontsize)
                    f.subplots_adjust(hspace=0)
                    #print(sig)
                    #print('min: {}, max: {}'.format(np.min(sig_arr), np.max(sig_arr)))
                ax = axarr[-1] 
                #         ax.semilogy((-truth+0.0001),label='ground truth')
                #         ax.plot(-prediction+0.0001,'g',label='neural net prediction')
                #         ax.axhline(-P_thresh_opt,color='k',label='trigger threshold')
        #         nn = np.min(pred)
        #        ax.plot(xx,truth,'g',label='target',linewidth=2)
        #         ax.axhline(0.4,linestyle="--",color='k',label='threshold')
                print('predictions shape:',prediction.shape)
                print('truth shape:',truth.shape)
                
     #           prediction=prediction[:,0]
                prediction=prediction#-1.5
                prediction[prediction<0]=0.0
                minii,maxii= np.amin(prediction),np.amax(prediction)
                lm_plot_max=np.amax(lm_plot)
                lm_plot=lm_plot/lm_plot_max*maxii
                truth_plot_max=np.amax(truth[:,1])
                truth_plot=truth[:,1]/truth_plot_max*maxii

                print('******************************************************')
                print('******************************************************')
                print('******************************************************')
                print('Truth_plot',truth_plot[:-10])
                print('lm_plot',lm_plot[:-10])
                print('******************************************************')
                target_plot_max=np.amax(target_plot)
                target_plot=target_plot/target_plot_max*maxii
               # ax.plot(xx,truth_plot,'yellow',label='truth')
                ax.plot(xx,lm_plot,'pink',label='Locked mode amplitude')
                ax.plot(xx,target_plot,'cyan',label='n1rms')
                #ax.plot(xx,truth,'pink',label='target')
                ax.plot(xx,prediction[:,0],'blue',label='FRNN-U predicted n=1 mode ',linewidth=2)
                ax.plot(xx,prediction[:,1],'red',label='FRNN-U predicted locked mode ',linewidth=2)
                #ax.axhline(P_thresh_opt,linestyle="--",color='k',label='threshold',zorder=2)
                #ax.axvline(p0,linestyle='--',color='darkgrey')
    #            ax.set_ylim([np.amin(prediction,truth),np.amax(prediction,truth)])
                ax.set_ylim([0,maxii])
                print('predictions:',shot.number,prediction)
#                ax.set_ylim([np.min([prediction,target_plot,lm_plot]),np.max([prediction,target_plot,lm_plot])])
                #ax.set_yticks([0,1])
                #if p0>0:
                #  ax.scatter(xx[k],p,s=300,marker='*',color='r',zorder=3)
                
                # if len(truth)-T_max_warn >= 0:
                #     ax.axvline(len(truth)-T_max_warn,color='r')#,label='max warning time')
        #        ax.axvline(len(truth)-self.T_min_warn,color='r',linewidth=0.5)#,label='min warning time')
                ax.set_xlabel('T [ms]',size=fontsize)
                # ax.axvline(2400)
                ax.legend(loc="center left",labelspacing=0.1,bbox_to_anchor=(1,0.5),fontsize=fontsize+2,frameon=False)
                plt.setp(ax.get_yticklabels(),fontsize=fontsize)
                plt.setp(ax.get_xticklabels(),fontsize=fontsize)
                # plt.xlim(0,200)
                plt.xlim([lower_lim,len(truth)])
        #         plt.savefig("{}.png".format(num),dpi=200,bbox_inches="tight")
                if save_fig:
                    plt.savefig('sig_fig_{}{}.png'.format(shot.number,extra_filename),bbox_inches='tight')
                    #np.savez('sig_{}{}.npz'.format(shot.number,extra_filename),shot=shot,T_min_warn=self.T_min_warn,T_max_warn=self.T_max_warn,prediction=prediction,truth=truth,use_signals=use_signals,P_thresh=P_thresh_opt)
                #plt.show()
        else:
            print("Shot hasn't been processed")



    def plot_shot_old(self,shot,save_fig=True,normalize=True,truth=None,prediction=None,P_thresh_opt=None,prediction_type='',extra_filename=''):
        if self.normalizer is None and normalize:
            if self.conf is not None:
                self.saved_conf['paths']['normalizer_path'] = self.conf['paths']['normalizer_path']
            nn = Normalizer(self.saved_conf)
            nn.train()
            self.normalizer = nn
            self.normalizer.set_inference_mode(True)

        if(shot.previously_saved(self.shots_dir)):
            shot.restore(self.shots_dir)
            t_disrupt = shot.t_disrupt
            is_disruptive =  shot.is_disruptive
            if normalize:
                self.normalizer.apply(shot)

            use_signals = self.saved_conf['paths']['use_signals']
            f,axarr = plt.subplots(len(use_signals)+1,1,sharex=Falsee,figsize=(13,13))#, squeeze=False)
            #plt.title(prediction_type)
            #all files must agree on T_warning due to output of truth vs. normalized shot ttd.
            assert(np.all(shot.ttd.flatten() == truth.flatten()))
            for i,sig in enumerate(use_signals):
                num_channels = sig.num_channels
                ax = axarr[i]
                sig_arr = shot.signals_dict[sig]
                
                if num_channels == 1:
                    ax.plot(sig_arr[:,0],label = sig.description)
                else:
                    ax.imshow(sig_arr[:,:].T, aspect='auto', label = sig.description + " (profile)")
                    ax.set_ylim([0,num_channels])
                ax.legend(loc='best',fontsize=8)
                plt.setp(ax.get_xticklabels(),visible=False)
                plt.setp(ax.get_yticklabels(),fontsize=7)
                f.subplots_adjust(hspace=0)
                #print(sig)
                #print('min: {}, max: {}'.format(np.min(sig_arr), np.max(sig_arr)))
                ax = axarr[-1] 
            if self.pred_ttd:
                ax.semilogy((-truth+0.0001),label='ground truth')
                ax.plot(-prediction+0.0001,'g',label='neural net prediction')
                ax.axhline(-P_thresh_opt,color='k',label='trigger threshold')
            else:
                ax.plot((truth+0.001),label='ground truth')
                ax.plot(prediction,'g',label='neural net prediction')
                ax.axhline(P_thresh_opt,color='k',label='trigger threshold')
            #ax.set_ylim([1e-5,1.1e0])
            ax.set_ylim([-2,2])
            if len(truth)-self.T_max_warn >= 0:
                ax.axvline(len(truth)-self.T_max_warn,color='r',label='min warning time')
            ax.axvline(len(truth)-self.T_min_warn,color='r',label='max warning time')
            ax.set_xlabel('T [ms]')
            #ax.legend(loc = 'lower left',fontsize=10)
            plt.setp(ax.get_yticklabels(),fontsize=7)
            # ax.grid()           
            if save_fig:
                plt.savefig('sig_fig_{}{}.png'.format(shot.number,extra_filename),bbox_inches='tight')
               # np.savez('sig_{}{}.npz'.format(shot.number,extra_filename),shot=shot,T_min_warn=self.T_min_warn,T_max_warn=self.T_max_warn,prediction=prediction,truth=truth,use_signals=use_signals,P_thresh=P_thresh_opt)
            plt.close()
        else:
            print("Shot hasn't been processed")


    def tradeoff_plot(self,accuracy_range,missed_range,fp_range,early_alarm_range,save_figure=False,plot_string='',linestyle="-",thres_opt=None):
        fontsize=15
        plt.figure()
        P_thresh_range = self.get_p_thresh_range()
        # semilogx(P_thresh_range,accuracy_range,label="accuracy")
        if self.pred_ttd:
            plt.semilogx(abs(P_thresh_range[::-1]),missed_range,'r',label="missed",linestyle=linestyle)
            plt.plot(abs(P_thresh_range[::-1]),fp_range,'k',label="false positives",linestyle=linestyle)
        else:
            plt.plot(P_thresh_range,missed_range,'r',label="missed",linestyle=linestyle)
            plt.plot(P_thresh_range,fp_range,'k',label="false positives",linestyle=linestyle)
        # plot(P_thresh_range,early_alarm_range,'c',label="early alarms")
        plt.legend(loc=(1.0,.6))
        plt.xlabel('Alarm threshold',size=fontsize)
        plt.grid()
        title_str = 'metrics{}'.format(plot_string.replace('_',' '))
        plt.title(title_str)
        if save_figure:
            plt.savefig(title_str + '.png',bbox_inches='tight')
        plt.close('all')
        plt.plot(fp_range,1-missed_range,'-b',linestyle=linestyle)
        if thres_opt!=None:
            idx_opt=(np.abs(P_thresh_range-thres_opt)).argmin()
            fp_opt=fp_range[idx_opt]
            tp_opt=1-missed_range[idx_opt]
            plt.plot(fp_opt,tp_opt,marker='o')
        ax = plt.gca()
        plt.xlabel('FP rate',size=fontsize)
        plt.ylabel('TP rate',size=fontsize)
        major_ticks = np.arange(0,1.01,0.2)
        minor_ticks = np.arange(0,1.01,0.05)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.set_xticks(minor_ticks,minor=True)
        ax.set_yticks(minor_ticks,minor=True)
        plt.setp(plt.gca().get_yticklabels(),fontsize=fontsize)
        plt.setp(plt.gca().get_xticklabels(),fontsize=fontsize)
        ax.grid(which='both')
        ax.grid(which='major',alpha=0.5)
        ax.grid(which='minor',alpha=0.3)
        plt.xlim([0,1])
        plt.ylim([0,1])
        if save_figure:
            plt.savefig(title_str + '_roc.png',bbox_inches='tight',dpi=200)
        print('ROC area ({}) is {}'.format(plot_string,self.roc_from_missed_fp(missed_range,fp_range)))
        return P_thresh_range,missed_range,fp_range

    def get_pred_truth_disr_by_shot(self,shot):
        if shot in self.shot_list_test:
            mode = 'test'
        elif shot in self.shot_list_train:
            mode = 'train'
        else:
            print('Shot {} not found'.format(shot))
            exit(1)
        if mode == 'test':
            pred = self.pred_test
            truth = self.truth_test
            is_disruptive = self.disruptive_test
            shot_list = self.shot_list_test
        else:
            pred = self.pred_train
            truth = self.truth_train
            is_disruptive = self.disruptive_train
            shot_list = self.shot_list_train
        i = shot_list.index(shot)
        t = truth[i]
        p = pred[i]
        is_disr = is_disruptive[i]
        shot = shot_list.shots[i]
        return p,t,is_disr

    def save_shot(self,shot,P_thresh_opt = 0,extra_filename=''):
        if self.normalizer is None:
            if self.conf is not None:
                self.saved_conf['paths']['normalizer_path'] = self.conf['paths']['normalizer_path']
            nn = Normalizer(self.saved_conf)
            nn.train()
            self.normalizer = nn
            self.normalizer.set_inference_mode(True)
 
        shot.restore(self.shots_dir)
        t_disrupt = shot.t_disrupt
        is_disruptive =  shot.is_disruptive
        self.normalizer.apply(shot)



        pred,truth,is_disr = self.get_pred_truth_disr_by_shot(shot)
        use_signals = self.saved_conf['paths']['use_signals']
        np.savez('sig_{}{}.npz'.format(shot.number,extra_filename),shot=shot,T_min_warn=self.T_min_warn,T_max_warn=self.T_max_warn,prediction=pred,truth=truth,use_signals=use_signals,P_thresh=P_thresh_opt)

    def get_roc_area_by_mode(self,mode='test'):
        if mode == 'test':
            pred = self.pred_test
            truth = self.truth_test
            is_disruptive = self.disruptive_test
            shot_list = self.shot_list_test
        else:
            pred = self.pred_train
            truth = self.truth_train
            is_disruptive = self.disruptive_train
            shot_list = self.shot_list_train
        return self.get_roc_area(pred,truth,is_disruptive)

    def get_roc_area(self,all_preds,all_truths,all_disruptive):
        correct_range, accuracy_range, fp_range,missed_range,early_alarm_range = \
         self.get_metrics_vs_p_thresh_custom(all_preds,all_truths,all_disruptive)

        return self.roc_from_missed_fp(missed_range,fp_range)

    def roc_from_missed_fp(self,missed_range,fp_range):
        #print(fp_range)
        #print(missed_range)
        return -np.trapz(1-missed_range,x=fp_range)


import os,sys
import numpy as np

from conf import conf

#mode = 'test'
file_num = 0
save_figure = True
pred_ttd = False

# cut_shot_ends = conf['data']['cut_shot_ends']
# dt = conf['data']['dt']
# T_max_warn = int(round(conf['data']['T_warning']/dt))
# T_min_warn = conf['data']['T_min_warn']#int(round(conf['data']['T_min_warn']/dt))
# if cut_shot_ends:
# 	T_max_warn = T_max_warn-T_min_warn
# 	T_min_warn = 0
T_min_warn = 30 #None #take value from conf #30

verbose=False
if len(sys.argv) > 1:
    results_dir = sys.argv[1]
else:
    results_dir = conf['paths']['results_prepath']
shots_dir = conf['paths']['processed_prepath']
print(results_dir)
analyzer = PerformanceAnalyzer(conf=conf,results_dir=results_dir,shots_dir=shots_dir,i = file_num,
T_min_warn = T_min_warn, verbose = verbose, pred_ttd=pred_ttd) 
P_thresh_opt = 0
analyzer.load_ith_file()

normalize = True
#analyzer.example_plots(P_thresh_opt,'test',['TP'],max_plot=10,extra_filename='test_D',normalize=normalize)
#analyzer.example_plots(P_thresh_opt,'test',['TN'],max_plot=30,extra_filename='test_ND',normalize=normalize)
#analyzer.example_plots(P_thresh_opt,'test',['FP'],max_plot=30,extra_filename='test_ND',normalize=normalize)
#analyzer.example_plots(P_thresh_opt,'test',['FN'],max_plot=30,extra_filename='test_D',normalize=normalize)

#P_thresh_opt = analyzer.compute_tradeoffs_and_print_from_training()
#P_thresh_opt = 0.566#0.566#0.92# analyzer.compute_tradeoffs_and_print_from_training()
linestyle="-"

#P_thresh_range,missed_range,fp_range = analyzer.compute_tradeoffs_and_plot('test',save_figure=save_figure,plot_string='_test',linestyle=linestyle,thres_opt=P_thresh_opt)

#ianalyzer.check_modes('test',bar_plot = True)

#np.savez('test_roc.npz',"P_thresh_range",P_thresh_range,"missed_range",missed_range,"fp_range",fp_range)
#P_thresh_range,missed_range,fp_range=analyzer.compute_tradeoffs_and_plot('train',save_figure=save_figure,plot_string='_train',linestyle=linestyle,thres_opt=P_thresh_opt)
#np.savez('train_roc.npz',"P_thresh_range",P_thresh_range,"missed_range",missed_range,"fp_range",fp_range)

#nalyzer.summarize_shot_prediction_stats_by_mode(P_thresh_opt,'test')
P_thresh_opt=0
normalize = True
yprime = analyzer.pred_test
ygold = analyzer.truth_test
print(yprime.shape)
print(ygold.shape)
#from performance_nov20 import * #PerformanceAnalyzer
#analyzer = PerformanceAnalyzer(conf=conf,results_dir=results_dir,shots_dir=shots_dir,i = file_num,
#T_min_warn = T_min_warn, verbose = verbose, pred_ttd=pred_ttd) 
#P_thresh_opt = 0
#analyzer.load_ith_file()
analyzer.check_modes()
thresh_range_t,TP,FP,roc_area,TP_re, roc_area_re = analyzer.calculate_sig_roc(yprime = yprime,ygold=ygold,pred_index = 1, truth_positive_threshold = 10.59)
#thresh_range_t,TP,FP,roc_area= analyzer.calculate_sig_roc(yprime = yprime,ygold=ygold,pred_index = 0, truth_positive_threshold = 12.23)
#print('thresh_range',thresh_range)
#print('TP',TP)
#print('FP',FP)
print('TP_length',len(TP))
print('FP_length',len(FP))
print('ROC',roc_area)
print('ROC_relax',roc_area_re)
plotting = True
##########plotting:
if plotting == True:
  plt.figure()
  plt.plot(FP,TP,color = 'black')
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.xlim([0,1])
  plt.ylim([0,1])
  plt.grid(linestyle = '--',color = 'grey')
  plt.savefig('ROC_curve.png')

  plt.figure()
  plt.plot(FP,TP_re,color = 'black')
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.xlim([0,1])
  plt.ylim([0,1])
  plt.grid(linestyle = '--',color = 'grey')

  plt.savefig('ROC_curve_relax.png')


