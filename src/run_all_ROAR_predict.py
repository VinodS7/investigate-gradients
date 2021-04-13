import subprocess
import os
import numpy as np

from argparse import ArgumentParser

def opts_parser():
    descr = "Runs all training for Remove and retrain"
    parser = ArgumentParser(description=descr)
    parser.add_argument('--modelpath', metavar='MODELPATH',
            nargs='+', type=str,
            help='Path to save models and data')
    parser.add_argument('--loss-grad-path', metavar='GRADPATH',
            nargs='+', type=str,
            help='Path to save loss gradients')
    parser.add_argument('--cache-spectra', metavar='DIR',
            type=str, default=None,
            help='Store pr retrieve spectra from directory')
    parser.add_argument('--attribution-method',type=str,
            nargs='+', default='grad',
            help='Choose algorithm for doing the'
            'loss gradient analysis')
    parser.add_argument('--save-distribution', type=int, default=0,
            help='Determine whether to save the occluded distribution or not')
    return parser

def main():
    parser = opts_parser()
    options = parser.parse_args()
    modelpath = options.modelpath
    lossgradpath = options.loss_grad_path
    num = ((np.arange(0,1,0.2)+0.1)*115*80).astype(np.int)
    num = np.append(num, 9100)
    input(num)
    replacement = ['mean']
    for m in range(len(modelpath)):
        path = modelpath[m]
        for att in options.attribution_method:
            print(att)
            for i in range(len(num)):
                for r in replacement:
                    print(r)
                    if(not os.path.exists(path+"jamendo_preds_occlude_ROAR_" +
                        str(att)+"_" 
                            +str(num[i])+"_"+r+".npz")):
                        print('computing preds: ',path+"jamendo_preds_occlude_ROAR_" +
                            str(att)+"_"+str(num[i])+"_"+r+".npz")
                        call = ["python", "src/ROAR_predict.py", path +"model.pth",
                            path + "jamendo_preds_occlude_ROAR_" + str(att)+"_"+str(num[i])+"_"+r+".npz",
                            "--cache-spectra",options.cache_spectra,"--ROAR", str(1),
                            "--input_type", "mel_spects", "--var", "occlude="+str(num[i]),
                            "--loss-grad-save", lossgradpath[m], '--lossgradient', path + "model.pth",
                            "--attribution-method", str(att), "--replacement-type", r]
                        subprocess.run(call)
                     
                    print("ROAR, occlusion: "+str(num[i]))
                    call = ["python", "src/eval.py", path+"jamendo_preds_occlude_ROAR_"+
                        str(att)+"_"+str(num[i])+"_"+r+".npz","--auroc"]
                    subprocess.run(call)
                    if(not os.path.exists(path+"jamendo_preds_occlude_KAR_" +
                        str(att)+"_"+str(num[i])+"_"+r+".npz")):
                        print('computing preds: ',path+"jamendo_preds_occlude_ROAR_" +
                            str(att)+"_" 
                            +str(num[i])+"_"+r+".npz")
                        call = ["python", "src/ROAR_predict.py", path +"model.pth",
                            path + "jamendo_preds_occlude_KAR_" + str(att)+"_"+str(num[i])+"_"+r+".npz",
                            "--cache-spectra",options.cache_spectra,"--ROAR", str(0),
                            "--input_type", "mel_spects", "--var", "occlude="+str(num[i]),
                            "--loss-grad-save", lossgradpath[m], '--lossgradient', path + "model.pth",
                            "--attribution-method", str(att), "--replacement-type", r]
            
                        subprocess.run(call)
                    print("KAR, occlusion: "+str(num[i]))
                
                    call = ["python", "src/eval.py", path+"jamendo_preds_occlude_KAR_"+
                        str(att)+"_"+str(num[i])+"_"+r+".npz","--auroc"]
                    subprocess.run(call)
                


if __name__=='__main__':
    main()
