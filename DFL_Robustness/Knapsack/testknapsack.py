import argparse
from argparse import Namespace
import pytorch_lightning as pl
import shutil
import random
import os 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from DPO import fenchel_young as fy
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from Trainer.PO_models import *
from Trainer.data_utils import KnapsackDataModule
from pytorch_lightning import loggers as pl_loggers
from distutils.util import strtobool
from Trainer.comb_solver import knapsack_solver, cvx_knapsack_solver,  intopt_knapsack_solver
from Trainer.diff_layer import SPOlayer, DBBlayer
from Trainer.CacheLosses import *
from imle.wrapper import imle
from imle.target import TargetDistribution
from imle.noise import SumOfGammaNoiseDistribution
#from Trainer.utils import batch_solve#, regret_fn, abs_regret_fn, regret_list,  growpool_fn


parser = argparse.ArgumentParser()
parser.add_argument("--capacity", type=int, help="capacity of knapsack", default= 12, required= True)
parser.add_argument("--model", type=str, help="name of the model", default= "", required= True)
parser.add_argument("--loss", type= str, help="loss", default= "", required=False)

parser.add_argument("--lambda_val", type=float, help="interpolaton parameter blackbox", default= 1., required=False)
parser.add_argument("--sigma", type=float, help="DPO FY noise parameter", default= 1., required=False)
parser.add_argument("--num_samples", type=int, help="number of samples FY", default= 1, required=False)

parser.add_argument("--temperature", type=float, help="input and target noise temperature parameter", default= 1., required=False)
parser.add_argument("--nb_iterations", type=int, help="number of iterations", default= 10, required=False)
parser.add_argument("--k", type=int, help="parameter k", default= 10, required=False)
parser.add_argument("--nb_samples", type=int, help="Number of samples paprameter", default= 10, required=False)
parser.add_argument("--beta", type=float, help="parameter lambda of IMLE", default= 10., required=False)


parser.add_argument("--mu", type=float, help="Regularization parameter DCOL & QPTL", default= 10., required=False)
parser.add_argument("--thr", type=float, help="threshold parameter", default= 1e-6)
parser.add_argument("--damping", type=float, help="damping parameter", default= 1e-8)
parser.add_argument("--tau", type=float, help="parameter of rankwise losses", default= 1e-8)
parser.add_argument("--growth", type=float, help="growth parameter of rankwise losses", default= 1e-8)
#parser.add_argument("--eps", type=float, help="adv pertubation", default= 0.0, required=False)

parser.add_argument("--lr", type=float, help="learning rate", default= 1e-3, required=False)
parser.add_argument("--batch_size", type=int, help="batch size", default= 128, required=False)
parser.add_argument("--max_epochs", type=int, help="maximum number of epochs", default= 35, required=False)
parser.add_argument("--num_workers", type=int, help="maximum number of workers", default= 4, required=False)

parser.add_argument('--scheduler', dest='scheduler',  type=lambda x: bool(strtobool(x)))
#eps = 0.2


if __name__ == "__main__":
    args = parser.parse_args()

    class _Sentinel:
        pass
    sentinel = _Sentinel()

    argument_dict = vars(args)
    get_class = lambda x: globals()[x]
    modelcls = get_class(argument_dict['model'])
    modelname = argument_dict.pop('model')
    capacity= argument_dict.pop('capacity')
    
    #lr_= argument_dict.pop('lr')
    #mu_= argument_dict.pop('mu')
    #print(lr_,mu_)
    
    sentinel_ns = Namespace(**{key:sentinel for key in argument_dict})
    parser.parse_args(namespace=sentinel_ns)

    explicit = {key:value for key, value in vars(sentinel_ns).items() if value is not sentinel }
    
    


    torch.use_deterministic_algorithms(True)
    def seed_all(seed):
        print("[ Using Seed : ", seed, " ]")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    ##########YF#######################
    
    def fgsm_attack(x, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        perturbed_x = x + epsilon*sign_data_grad
        return perturbed_x
    
    
    def DCOLgenerate_adv_dataloader(weights,capacity,n_items,model,test_dataloader,eps):
        comblayer = cvx_knapsack_solver(weights,capacity,n_items,mu=args.mu)
        adv_x_list = []
        x_list = []
        y_list = []
        sol_list = []
        i=0
        for test_batch in test_dataloader:
            # Send the data and label to the device
            
            
            x,y,sol = test_batch
            x.requires_grad =  True
           # y.requires_grad =  True
           # sol.requires_grad =  True
            y_hat =  model(x).squeeze()
            sol_hat = comblayer(y_hat)
            
            
            loss = ((sol - sol_hat)*y).sum(-1).mean()
            model.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            adv_x = fgsm_attack(x, eps, data_grad)
            adv_x_list.append(adv_x.detach())
            print(adv_x.requires_grad,x.requires_grad,y.requires_grad,sol.requires_grad)
            x_list.append(x.detach())
            y_list.append(y)
            sol_list.append(sol)
            
        adv_x_tensor = torch.cat((adv_x_list[0], adv_x_list[1]), dim=0)
        x_tensor = torch.cat((x_list[0], x_list[1]), dim=0)
        y_tensor = torch.cat((y_list[0], y_list[1]), dim=0)
        sol_tensor = torch.cat((sol_list[0], sol_list[1]), dim=0)

        # Create a new DataLoader with adv_x, y, and sol
        adv_dataset = TensorDataset(adv_x_tensor, x_tensor, y_tensor, sol_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    
    
    
    def generate_adv_dataloader_MSE(weights,capacity,n_items,model,test_dataloader,eps):
        lossfn = nn.MSELoss()
        #comblayer = cvx_knapsack_solver(weights,capacity,n_items,mu=args.mu)
        adv_x_list = []
        x_list = []
        y_list = []
        sol_list = []
        i=0
        for test_batch in test_dataloader:
            # Send the data and label to the device
            
            
            x,y,sol = test_batch
            x.requires_grad =  True
           # y.requires_grad =  True
           # sol.requires_grad =  True
            
            y_hat =  model(x).squeeze()
            
            
            loss = lossfn(y_hat,y)#((sol - sol_hat)*y).sum(-1).mean()
           
            model.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            adv_x = fgsm_attack(x, eps, data_grad)
            adv_x_list.append(adv_x.detach())
            x_list.append(x.detach())
            y_list.append(y)
            sol_list.append(sol)
           
        adv_x_tensor = torch.cat((adv_x_list[0], adv_x_list[1]), dim=0)
        x_tensor = torch.cat((x_list[0], x_list[1]), dim=0)
        y_tensor = torch.cat((y_list[0], y_list[1]), dim=0)
        sol_tensor = torch.cat((sol_list[0], sol_list[1]), dim=0)

        # Create a new DataLoader with adv_x, y, and sol
        adv_dataset = TensorDataset(adv_x_tensor,x_tensor ,y_tensor, sol_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    def SPOgenerate_adv_dataloader(weights,capacity,n_items,model,test_dataloader,eps):
        solver = knapsack_solver(weights,capacity,n_items)
        layer = SPOlayer(solver)
        adv_x_list = []
        x_list = []
        y_list = []
        sol_list = []
        i=0
        for test_batch in test_dataloader:
            # Send the data and label to the device
            
            
            x,y,sol = test_batch
            x.requires_grad =  True
           # y.requires_grad =  True
           # sol.requires_grad =  True
            #y_hat =  model(x).squeeze()
            #sol_hat = comblayer(y_hat)
            
            y_hat =  model(x).squeeze()
            loss =  layer(y_hat, y, sol) 
            

            model.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            adv_x = fgsm_attack(x, eps, data_grad)
            adv_x_list.append(adv_x.detach())
            x_list.append(x.detach())
            y_list.append(y)
            sol_list.append(sol)
           
        adv_x_tensor = torch.cat((adv_x_list[0], adv_x_list[1]), dim=0)
        x_tensor = torch.cat((x_list[0], x_list[1]), dim=0)
        y_tensor = torch.cat((y_list[0], y_list[1]), dim=0)
        sol_tensor = torch.cat((sol_list[0], sol_list[1]), dim=0)

        # Create a new DataLoader with adv_x, y, and sol
        adv_dataset = TensorDataset(adv_x_tensor, x_tensor,y_tensor, sol_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    
    def DBBgenerate_adv_dataloader(weights,capacity,n_items,model,test_dataloader,eps):
        solver = knapsack_solver(weights,capacity,n_items)
        layer = DBBlayer(solver, lambda_val=args.lambda_val)
        adv_x_list = []
        x_list = []
        y_list = []
        sol_list = []
        i=0
        for test_batch in test_dataloader:
            # Send the data and label to the device
            
            
            x,y,sol = test_batch
            x.requires_grad =  True
        
             
            y_hat =  model(x).squeeze()
            sol_hat = layer(y_hat, y,sol ) 
            loss = ((sol - sol_hat)*y).sum(-1).mean()
            

            model.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            adv_x = fgsm_attack(x, eps, data_grad)
            adv_x_list.append(adv_x.detach())
            x_list.append(x.detach())
            y_list.append(y)
            sol_list.append(sol)
           
        adv_x_tensor = torch.cat((adv_x_list[0], adv_x_list[1]), dim=0)
        x_tensor = torch.cat((x_list[0], x_list[1]), dim=0)
        y_tensor = torch.cat((y_list[0], y_list[1]), dim=0)
        sol_tensor = torch.cat((sol_list[0], sol_list[1]), dim=0)

        # Create a new DataLoader with adv_x, y, and sol
        adv_dataset = TensorDataset(adv_x_tensor, x_tensor, y_tensor, sol_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    
    def IMLE_generate_adv_dataloader(weights,capacity,n_items,model,test_dataloader,eps):
        #solver = knapsack_solver(weights,capacity,n_items)
        solver = knapsack_solver(weights,capacity, n_items)
        #layer = DBBlayer(solver, lambda_val=args.lambda_val)
        
        imle_solver = lambda y_: batch_solve(solver,y_)

        target_distribution = TargetDistribution(alpha=1.0, beta=args.beta)
        noise_distribution = SumOfGammaNoiseDistribution(k= args.k, nb_iterations= args.nb_iterations)

        layer = imle(imle_solver,  target_distribution=target_distribution,noise_distribution=noise_distribution,
                    input_noise_temperature= args.temperature, target_noise_temperature= args.temperature,
                    nb_samples= args.nb_samples)
       
        adv_x_list = []
        x_list = []
        y_list = []
        sol_list = []
        i=0
        for test_batch in test_dataloader:
            # Send the data and label to the device
            
            
            x,y,sol = test_batch
            x.requires_grad =  True
        
             
            y_hat =  model(x).squeeze()
            sol_hat = layer(y_hat ) 
            loss = ((sol - sol_hat)*y).sum(-1).mean() 
            

            model.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            adv_x = fgsm_attack(x, eps, data_grad)
            adv_x_list.append(adv_x.detach())
            x_list.append(x.detach())
            y_list.append(y)
            sol_list.append(sol)
          
        adv_x_tensor = torch.cat((adv_x_list[0], adv_x_list[1]), dim=0)
        x_tensor = torch.cat((x_list[0], x_list[1]), dim=0)
        y_tensor = torch.cat((y_list[0], y_list[1]), dim=0)
        sol_tensor = torch.cat((sol_list[0], sol_list[1]), dim=0)

        # Create a new DataLoader with adv_x, y, and sol
        adv_dataset = TensorDataset(adv_x_tensor,x_tensor ,y_tensor, sol_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    
    
    def FYgenerate_adv_dataloader(weights,capacity,n_items,model,test_dataloader,eps):
        solver = knapsack_solver(weights,capacity,n_items)
        #layer = DBBlayer(solver, lambda_val=args.lambda_val)
        fy_solver =  lambda y_: batch_solve(solver,y_)
        criterion = fy.FenchelYoungLoss(fy_solver, num_samples= args.num_samples, 
        sigma= args.sigma,maximize = True, batched= True)
        
        adv_x_list = []
        x_list = []
        y_list = []
        sol_list = []
        for test_batch in test_dataloader:
            # Send the data and label to the device
            
            
            x,y,sol = test_batch
            x.requires_grad =  True
        
            y_hat =  model(x).squeeze()
            loss = criterion(y_hat,sol).mean()
           
            

            model.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            adv_x = fgsm_attack(x, eps, data_grad)
            adv_x_list.append(adv_x.detach())
            x_list.append(x.detach())
            y_list.append(y)
            sol_list.append(sol)
           
        adv_x_tensor = torch.cat((adv_x_list[0], adv_x_list[1]), dim=0)
        x_tensor = torch.cat((x_list[0], x_list[1]), dim=0)
        y_tensor = torch.cat((y_list[0], y_list[1]), dim=0)
        sol_tensor = torch.cat((sol_list[0], sol_list[1]), dim=0)

        # Create a new DataLoader with adv_x, y, and sol
        adv_dataset = TensorDataset(adv_x_tensor,x_tensor ,y_tensor, sol_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    
    
    def IntOptgenerate_adv_dataloader(weights,capacity,n_items,model,test_dataloader,eps):
        comblayer = intopt_knapsack_solver(weights,capacity,n_items, thr= args.thr,damping= args.damping)
        
        
        adv_x_list = []
        x_list = []
        y_list = []
        sol_list = []
        i=0
        for test_batch in test_dataloader:
            # Send the data and label to the device
            
            
            x,y,sol = test_batch
            x.requires_grad =  True
        
            y_hat =  model(x).squeeze()
            sol_hat = comblayer(y_hat)
            loss = ((sol - sol_hat)*y).sum(-1).mean()
           
            model.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            adv_x = fgsm_attack(x, eps, data_grad)
            adv_x_list.append(adv_x.detach())
            x_list.append(x.detach())
            y_list.append(y)
            sol_list.append(sol)
           
        adv_x_tensor = torch.cat((adv_x_list[0], adv_x_list[1]), dim=0)
        x_tensor = torch.cat((x_list[0], x_list[1]), dim=0)
        y_tensor = torch.cat((y_list[0], y_list[1]), dim=0)
        sol_tensor = torch.cat((sol_list[0], sol_list[1]), dim=0)

        # Create a new DataLoader with adv_x, y, and sol
        adv_dataset = TensorDataset(adv_x_tensor,x_tensor, y_tensor, sol_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
   
    def Listwisegenerate_adv_dataloader(weights,capacity,n_items,init_cache,model,test_dataloader,eps,growth=0.1):
        #comblayer = intopt_knapsack_solver(weights,capacity,n_items, thr= args.thr,damping= args.damping)
        solver = knapsack_solver(weights,capacity, n_items)
        loss_fn = ListwiseLoss(temperature=args.tau)
        
        #growth = args.growth
        cache_np = init_cache.detach().numpy()
        cache_np = np.unique(cache_np,axis=0)
        # torch has no unique function, so we have to do this
        init_cache =  torch.from_numpy(cache_np).float()
        cache = init_cache
        
        adv_x_list = []
        x_list = []
        y_list = []
        sol_list = []
        i=0
        for test_batch in test_dataloader:
            # Send the data and label to the device
            
            
            x,y,sol = test_batch
      
            x.requires_grad =  True
            
            y_hat =  model(x).squeeze()
        
            if (np.random.random(1)[0]< growth) or len(cache)==0:
                cache= growpool_fn(solver,cache, y_hat)

            loss = loss_fn(y_hat,y,sol,cache)
        
           
            model.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            adv_x = fgsm_attack(x, eps, data_grad)
            adv_x_list.append(adv_x.detach())
            x_list.append(x.detach())
            y_list.append(y)
            sol_list.append(sol)
            
        adv_x_tensor = torch.cat((adv_x_list[0], adv_x_list[1]), dim=0)
        x_tensor = torch.cat((x_list[0], x_list[1]), dim=0)
        y_tensor = torch.cat((y_list[0], y_list[1]), dim=0)
        sol_tensor = torch.cat((sol_list[0], sol_list[1]), dim=0)

        # Create a new DataLoader with adv_x, y, and sol
        adv_dataset = TensorDataset(adv_x_tensor, x_tensor,y_tensor, sol_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    
    
    def Pairwisegenerate_adv_dataloader(weights,capacity,n_items,init_cache,model,test_dataloader,eps,growth=0.1):
        #comblayer = intopt_knapsack_solver(weights,capacity,n_items, thr= args.thr,damping= args.damping)
        solver = knapsack_solver(weights,capacity, n_items)
        loss_fn = PairwiseLoss(margin=args.tau)
        
       # growth = args.growth
        cache_np = init_cache.detach().numpy()
        cache_np = np.unique(cache_np,axis=0)
        # torch has no unique function, so we have to do this
        init_cache =  torch.from_numpy(cache_np).float()
        cache = init_cache
        
        adv_x_list = []
        x_list = []
        y_list = []
        sol_list = []
        
        for test_batch in test_dataloader:
            # Send the data and label to the device
            
            
            x,y,sol = test_batch
            #print("x/input",x)
            x.requires_grad =  True
            
            y_hat =  model(x).squeeze()
        
            if (np.random.random(1)[0]< growth) or len(cache)==0:
                cache= growpool_fn(solver,cache, y_hat)

            loss = loss_fn(y_hat,y,sol,cache)
        
           
            model.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            adv_x = fgsm_attack(x, eps, data_grad)
            adv_x_list.append(adv_x.detach())
            x_list.append(x.detach())
            y_list.append(y)
            sol_list.append(sol)
             

        adv_x_tensor = torch.cat((adv_x_list[0], adv_x_list[1]), dim=0)
        x_tensor = torch.cat((x_list[0], x_list[1]), dim=0)
        y_tensor = torch.cat((y_list[0], y_list[1]), dim=0)
        sol_tensor = torch.cat((sol_list[0], sol_list[1]), dim=0)

        # Create a new DataLoader with adv_x, y, and sol
        adv_dataset = TensorDataset(adv_x_tensor, x_tensor,y_tensor, sol_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    
    def PairwiseDiff_generate_adv_dataloader(weights,capacity,n_items,init_cache,model,test_dataloader,eps,growth=0.1):
        #comblayer = intopt_knapsack_solver(weights,capacity,n_items, thr= args.thr,damping= args.damping)
        solver = knapsack_solver(weights,capacity, n_items)
        loss_fn = PairwisediffLoss()
        
       # growth = args.growth
        cache_np = init_cache.detach().numpy()
        cache_np = np.unique(cache_np,axis=0)
        # torch has no unique function, so we have to do this
        init_cache =  torch.from_numpy(cache_np).float()
        cache = init_cache
        
        adv_x_list = []
        x_list = []
        y_list = []
        sol_list = []
        
        for test_batch in test_dataloader:
            # Send the data and label to the device
            
            
            x,y,sol = test_batch
            #print("x/input",x)
            x.requires_grad =  True
            
            y_hat =  model(x).squeeze()
        
            if (np.random.random(1)[0]< growth) or len(cache)==0:
                cache= growpool_fn(solver,cache, y_hat)

            loss = loss_fn(y_hat,y,sol,cache)
        
           
            model.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            adv_x = fgsm_attack(x, eps, data_grad)
            adv_x_list.append(adv_x.detach())
            x_list.append(x.detach())
            y_list.append(y)
            sol_list.append(sol)
             

        adv_x_tensor = torch.cat((adv_x_list[0], adv_x_list[1]), dim=0)
        x_tensor = torch.cat((x_list[0], x_list[1]), dim=0)
        y_tensor = torch.cat((y_list[0], y_list[1]), dim=0)
        sol_tensor = torch.cat((sol_list[0], sol_list[1]), dim=0)

        # Create a new DataLoader with adv_x, y, and sol
        adv_dataset = TensorDataset(adv_x_tensor,x_tensor ,y_tensor, sol_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    
    def MAP_generate_adv_dataloader(weights,capacity,n_items,init_cache,model,test_dataloader,eps,growth=0.1):
        #comblayer = intopt_knapsack_solver(weights,capacity,n_items, thr= args.thr,damping= args.damping)
        solver = knapsack_solver(weights,capacity, n_items)
        loss_fn = MAP()
        
       # growth = args.growth
        cache_np = init_cache.detach().numpy()
        cache_np = np.unique(cache_np,axis=0)
        # torch has no unique function, so we have to do this
        init_cache =  torch.from_numpy(cache_np).float()
        cache = init_cache
        
        adv_x_list = []
        x_list = []
        y_list = []
        sol_list = []
        
        for test_batch in test_dataloader:
            # Send the data and label to the device
            
            
            x,y,sol = test_batch
            #print("x/input",x)
            x.requires_grad =  True
            
            y_hat =  model(x).squeeze()
        
            if (np.random.random(1)[0]< growth) or len(cache)==0:
                cache= growpool_fn(solver,cache, y_hat)

            loss = loss_fn(y_hat,y,sol,cache)
        
           
            model.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            adv_x = fgsm_attack(x, eps, data_grad)
            adv_x_list.append(adv_x.detach())
            x_list.append(x.detach())
            y_list.append(y)
            sol_list.append(sol)
             

        adv_x_tensor = torch.cat((adv_x_list[0], adv_x_list[1]), dim=0)
        x_tensor = torch.cat((x_list[0], x_list[1]), dim=0)
        y_tensor = torch.cat((y_list[0], y_list[1]), dim=0)
        sol_tensor = torch.cat((sol_list[0], sol_list[1]), dim=0)

        # Create a new DataLoader with adv_x, y, and sol
        adv_dataset = TensorDataset(adv_x_tensor, x_tensor,y_tensor, sol_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
            
            
            
            
        
        
    ############YF########################
        


    ################## Define the outputfile
    outputfile = "Rslt/{}Knapsack{}.csv".format(modelname,args.loss)
    regretfile = "Rslt/{}KnapsackRegret{}.csv".format( modelname,args.loss )
    ckpt_dir =  "ckpt_dir/{}{}/".format( modelname,args.loss)
    log_dir = "lightning_logs/{}{}/".format(  modelname,args.loss )
    learning_curve_datafile = "LearningCurve/{}_".format(modelname)+"_".join( ["{}_{}".format(k,v) for k,v  in explicit.items()] )+".csv"  
    
    for directory in [os.path.dirname(outputfile), os.path.dirname(regretfile), ckpt_dir, log_dir, os.path.dirname(learning_curve_datafile)]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    shutil.rmtree(log_dir,ignore_errors=True)
    
    
    testresult_lst = []
    adv_test_result_lst = []
    MSEadv_test_result_lst = []

    for seed in range(10):  
        seed_all(seed)
        shutil.rmtree(ckpt_dir,ignore_errors=True)
        checkpoint_callback = ModelCheckpoint(
                #monitor="val_regret",mode="min",
                dirpath=ckpt_dir, 
                filename="model-{epoch:02d}-{val_regret:.8f}",
                )

        g = torch.Generator()
        g.manual_seed(seed)    
        data =  KnapsackDataModule(capacity=  capacity, batch_size= argument_dict['batch_size'], generator=g, seed= seed, num_workers= args.num_workers)
        weights, n_items =  data.weights, data.n_items
        if modelname=="CachingPO":
            cache = torch.from_numpy (data.train_df.sol)
            #########YF###########
            test_cache = torch.from_numpy (data.test_df.sol)
            ########YF##############
        tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)
        trainer = pl.Trainer(max_epochs= argument_dict['max_epochs'], 
        min_epochs=1,logger=tb_logger, callbacks=[checkpoint_callback])
        if modelname=="CachingPO":
            model =  modelcls(weights,capacity,n_items,init_cache=cache,seed=seed, **argument_dict)
        else:
            model =  modelcls(weights,capacity,n_items,seed=seed, **argument_dict)
        validresult = trainer.validate(model, datamodule=data)
        
        trainer.fit(model, datamodule=data)
        best_model_path = checkpoint_callback.best_model_path

        if modelname=="CachingPO":
            model =  modelcls.load_from_checkpoint(best_model_path,
            weights = weights,capacity= capacity,n_items = n_items,init_cache=cache,seed=seed, **argument_dict)
        else:
            model =  modelcls.load_from_checkpoint(best_model_path,
            weights = weights,capacity= capacity,n_items = n_items,seed=seed, **argument_dict)
    ##### SummaryWrite ######################
        
        eps_list = [0.01, 0.1, 0.15]
        for eps in eps_list:
                
            validresult = trainer.validate(model,datamodule=data)
            testresult = trainer.test(model, datamodule=data)
            print('test result dic',testresult)
            
            
            ######YF##########
            if modelname == "CachingPO" and args.loss == "MAP":
                adv_dataloader = MAP_generate_adv_dataloader(weights,capacity,n_items,test_cache,model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
                #adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

            
                MSEadv_dataloader = generate_adv_dataloader_MSE(weights,capacity,n_items,model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            
            
            if modelname == "CachingPO" and args.loss == "pairwise_diff":
                adv_dataloader = PairwiseDiff_generate_adv_dataloader(weights,capacity,n_items,test_cache,model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
            # adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

            
                MSEadv_dataloader = generate_adv_dataloader_MSE(weights,capacity,n_items,model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            
            
            if modelname == "CachingPO" and args.loss == "pairwise":
                adv_dataloader = Pairwisegenerate_adv_dataloader(weights,capacity,n_items,test_cache,model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
                #adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

            
                MSEadv_dataloader = generate_adv_dataloader_MSE(weights,capacity,n_items,model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            
            
            if modelname == "CachingPO" and args.loss == "listwise":
                adv_dataloader = Listwisegenerate_adv_dataloader(weights,capacity,n_items,test_cache,model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
            # adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

            
                MSEadv_dataloader = generate_adv_dataloader_MSE(weights,capacity,n_items,model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            
            
            if modelname == "IntOpt":
                adv_dataloader = IntOptgenerate_adv_dataloader(weights,capacity,n_items,model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
                #adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

            
                MSEadv_dataloader = generate_adv_dataloader_MSE(weights,capacity,n_items,model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            
            if modelname == "baseline_mse":
                #Unsure how fair of an assesment this is for baseline adv of DCOL
                adv_dataloader = DCOLgenerate_adv_dataloader(weights,capacity,n_items,model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
                #adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

            
                MSEadv_dataloader = generate_adv_dataloader_MSE(weights,capacity,n_items,model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            
            if modelname == "FenchelYoung":
                adv_dataloader = FYgenerate_adv_dataloader(weights,capacity,n_items,model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
            # adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

            
                MSEadv_dataloader = generate_adv_dataloader_MSE(weights,capacity,n_items,model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            
            
            if modelname == "DBB":
                adv_dataloader = DBBgenerate_adv_dataloader(weights,capacity,n_items,model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
            # adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

            
                MSEadv_dataloader = generate_adv_dataloader_MSE(weights,capacity,n_items,model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            
            if modelname == "SPO":
                adv_dataloader = SPOgenerate_adv_dataloader(weights,capacity,n_items,model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
                #adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

            
                MSEadv_dataloader = generate_adv_dataloader_MSE(weights,capacity,n_items,model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            if modelname == "DCOL":
                adv_dataloader = DCOLgenerate_adv_dataloader(weights,capacity,n_items,model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
                #adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

            
                MSEadv_dataloader = generate_adv_dataloader_MSE(weights,capacity,n_items,model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
                
            if modelname == "IMLE":
                adv_dataloader = IMLE_generate_adv_dataloader(weights,capacity,n_items,model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
            # adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

            
                MSEadv_dataloader = generate_adv_dataloader_MSE(weights,capacity,n_items,model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            

            
            
            
            testresult_lst.append(testresult[0])
            adv_test_result_lst.append(adv_testresult[0])
            MSEadv_test_result_lst.append( MSEadv_testresult[0])
            ########YF###########
            
            df = pd.DataFrame({**MSEadv_testresult[0],**adv_testresult[0], **testresult[0], **validresult[0]}, index=[0])
            for k,v in explicit.items():
                #print(k,v)
                df[k] = v
            df['seed'] = seed
            df['capacity'] = capacity
            df['epsilon'] = eps
            with open(outputfile, 'a') as f:
                    df.to_csv(f, header=f.tell()==0)


        regret_list = trainer.predict(model, data.test_dataloader())
        

        df = pd.DataFrame({"regret":regret_list[0].tolist()})
        df.index.name='instance'
        for k,v in explicit.items():
            df[k] = v
        df['seed'] = seed
        df['capacity'] =capacity    
        with open(regretfile, 'a') as f:
            df.to_csv(f, header=f.tell()==0)
    
    ######################YF######################## 
    #import matplotlib.pyplot as plt
    #import matplotlib.ticker as mtick
    #print("test result: ",testresult_lst)
    #print("adv test result", adv_test_result_lst)

    #print("test result type: ",type(testresult_lst))
    #print("adv test result  type: ", type(adv_test_result_lst))

    #print("test result [0]: ",testresult_lst[0])
    #print("adv test result [0]", adv_test_result_lst[0])
    
    #everything below is not  needed
    #test_regret_lst = []
    #adv_test_regret_lst = []
    #MSEadv_test_regret_lst = []
    
    
    #for i in testresult_lst: 
    #    test_regret_lst.append(i['test_regret'])
        
# for j in adv_test_result_lst:
    #   adv_test_regret_lst.append(j['adv_test_regret'])
    
    #for q in MSEadv_test_result_lst:
    #    MSEadv_test_regret_lst.append(q['MSEadv_test_regret'])
        

# data = {'test regret': test_regret_lst, 'adv test regret': adv_test_regret_lst, 'mse adv test regret':MSEadv_test_regret_lst }
# df = pd.DataFrame(data)

    # Specify the path where you want to save the .csv file
# csv_file_path = 'Rslt/DCOL_regret_data.csv'

    # Save the DataFrame to a .csv file
    #df.to_csv(csv_file_path, index=False)
        

    #plt.figure(figsize=(10, 6))

    
    #plt.boxplot(test_regret_lst, positions=[1], vert=True, labels=["{}".format(modelname)])
    


# plt.boxplot(adv_test_regret_lst, positions=[2], vert=True, labels=["{} adv".format(modelname)])
    
    #plt.boxplot(MSEadv_test_regret_lst, positions=[3], vert=True, labels=["{} MSE adv".format(modelname)])
    
    #plt.ylabel("test_regret (%)")
    #plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    
    # Set the title
    #plt.title("Comparison of test_regret on different attacks")

    # Show the plot
    #plt.show()
    ######################YF######################## 




    ###############################  Save  Learning Curve Data ########
    
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    parent_dir=   log_dir+"lightning_logs/"
    version_dirs = [os.path.join(parent_dir,v) for v in os.listdir(parent_dir)]

    walltimes = []
    steps = []
    regrets= []
    mses = []
    for logs in version_dirs:
        event_accumulator = EventAccumulator(logs)
        event_accumulator.Reload()

        events = event_accumulator.Scalars("val_regret")
        walltimes.extend( [x.wall_time for x in events])
        steps.extend([x.step for x in events])
        regrets.extend([x.value for x in events])
        events = event_accumulator.Scalars("val_mse")
        mses.extend([x.value for x in events])

    df = pd.DataFrame({"step": steps,'wall_time':walltimes,  "val_regret": regrets,
    "val_mse": mses })
    df['model'] = modelname
    df.to_csv(learning_curve_datafile)
        
        



        
