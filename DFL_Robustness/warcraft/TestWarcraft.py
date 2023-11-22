import argparse
from argparse import Namespace
from Trainer.data_utils import WarcraftDataModule, return_trainlabel, return_testlabel
from Trainer.Trainer import *
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch
import shutil
import random
from pytorch_lightning import loggers as pl_loggers
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pytorch_lightning.callbacks import ModelCheckpoint 

from comb_modules.losses import *
from Trainer.diff_layer import BlackboxDifflayer,SPOlayer, CvxDifflayer, IntoptDifflayer, QptDifflayer    
from torch.utils.data import DataLoader, TensorDataset
from imle.wrapper import imle
from imle.target import TargetDistribution
from imle.noise import SumOfGammaNoiseDistribution

parser = argparse.ArgumentParser()
parser.add_argument("--img_size", type=int, help="size of image in one dimension", default= 12)
parser.add_argument("--model", type=str, help="name of the model", default= "", required= True)
parser.add_argument("--loss", type= str, help="loss", default= "", required=False)


parser.add_argument("--lr", type=float, help="learning rate", default= 1e-3, required=False)
parser.add_argument("--batch_size", type=int, help="batch size", default= 128, required=False)
parser.add_argument("--max_epochs", type=int, help="maximum number of epochs", default= 70, required=False)
parser.add_argument("--seed", type=int, help="seed", default= 9, required=False)

parser.add_argument("--lambda_val", type=float, help="interpolaton parameter blackbox", default= 1., required=False)
parser.add_argument("--sigma", type=float, help="DPO FY noise parameter", default= 1., required=False)
parser.add_argument("--num_samples", type=int, help="number of samples FY", default= 1, required=False)

parser.add_argument("--temperature", type=float, help="input and target noise temperature parameter", default= 1., required=False)
parser.add_argument("--nb_iterations", type=int, help="number of iterations", default= 10, required=False)
parser.add_argument("--k", type=int, help="parameter k", default= 10, required=False)
parser.add_argument("--nb_samples", type=int, help="Number of samples paprameter", default= 10, required=False)
parser.add_argument("--beta", type=float, help="lambda parameter of IMLE", default= 10., required=False)
#parser.add_argument("--eps", type=float, help="adv pertubation", default= 0.0, required=False)


parser.add_argument("--mu", type=float, help="Regularization parameter DCOL & QPTL", default= 10., required=False)
parser.add_argument("--thr", type=float, help="threshold parameter", default= 1e-6)
parser.add_argument("--damping", type=float, help="damping parameter", default= 1e-8)
parser.add_argument("--tau", type=float, help="parameter of rankwise losses", default= 1e-8)
parser.add_argument("--growth", type=float, help="growth parameter of rankwise losses", default= 1e-8)



parser.add_argument("--output_tag", type=str, help="tag", default= 50, required=False)
parser.add_argument("--index", type=int, help="index", default= 1, required=False)
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
    img_size = argument_dict.pop('img_size')
    img_size = "{}x{}".format(img_size, img_size)
    #seed = argument_dict['seed']
    argument_dict.pop('output_tag')
    index = argument_dict.pop('index')
    argument_dict.pop('seed')
    #argument_dict.pop('seed')
    #argument_dict['seed'] = 77
    #print(**argument_dict)    

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
    ############### Configuration

    ####################YF

    def fgsm_attack(x, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
            sign_data_grad = data_grad.sign()
            perturbed_x = x + epsilon*sign_data_grad
            return perturbed_x
        
    def SPO_generate_adv_dataloader(model,test_dataloader,eps):
        neighbourhood_fn =  "8-grid"
        comb_layer =  SPOlayer( neighbourhood_fn= neighbourhood_fn)
        loss_fn = RegretLoss()
        
        adv_input_list = []
        input_list = []
        label_list = []
        true_weights_list = []
        for test_batch in test_dataloader:
            input, label, true_weights = test_batch
            input.requires_grad =  True
                
            output = model(input)
            weights = output.reshape(-1, output.shape[-1], output.shape[-1])
            
            ### For SPO, we need the true weights as we have to compute 2*\hat{c} - c
            shortest_path = comb_layer(weights, label, true_weights)
            loss = loss_fn(shortest_path, label,  true_weights)
            
            model.zero_grad()
            loss.backward()
            data_grad = input.grad.data
            adv_input = fgsm_attack(input, eps, data_grad)
            adv_input_list.append(adv_input.detach())
            input_list.append(input.detach())
            label_list.append(label)
            true_weights_list.append(true_weights)
            
        adv_input_tensor = torch.empty(0)    
        for i in range(len(adv_input_list)):
            adv_input_tensor = torch.cat((adv_input_tensor, adv_input_list[i]), dim=0)
            
        input_tensor = torch.empty(0)
        for i in range(len(input_list)):
            input_tensor = torch.cat((input_tensor, input_list[i]), dim=0)
        
        label_tensor = torch.empty(0)     
        for i in range(len(label_list)):
            label_tensor = torch.cat((label_tensor, label_list[i]), dim=0)
        true_weights_tensor = torch.empty(0) 
        for i in range(len(true_weights_list)):
            true_weights_tensor = torch.cat((true_weights_tensor, true_weights_list[i]), dim=0)
            
        adv_dataset = TensorDataset(adv_input_tensor,input_tensor ,label_tensor, true_weights_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader

    def MSE_generate_adv_dataloader(model,test_dataloader,eps):
        #neighbourhood_fn =  "8-grid"
        #comb_layer =  SPOlayer( neighbourhood_fn= neighbourhood_fn)
        #loss_fn = RegretLoss()
        criterion = nn.MSELoss(reduction='mean')
        
        adv_input_list = []
        label_list = []
        input_list = []
        true_weights_list = []
        for test_batch in test_dataloader:
            input, label, true_weights = test_batch
            input.requires_grad =  True
                
            output = model(input)
            #weights = output.reshape(-1, output.shape[-1], output.shape[-1])
            if len(output.size()) != 3:
                output = output.view(label.shape)
                #flat_weights = true_weights.view(true_weights.size()[0], -1).type_as(true_weights)
            #print(output.size(),flat_target.size())
            print("lossMSE dimensions",output.size(),true_weights.size())
            loss = criterion(output,true_weights).mean()
            model.zero_grad()
            loss.backward()
            data_grad = input.grad.data
            adv_input = fgsm_attack(input, eps, data_grad)
            adv_input_list.append(adv_input.detach())
            input_list.append(input.detach())
            label_list.append(label)
            true_weights_list.append(true_weights)
            
        adv_input_tensor = torch.empty(0)    
        for i in range(len(adv_input_list)):
            adv_input_tensor = torch.cat((adv_input_tensor, adv_input_list[i]), dim=0)
        input_tensor = torch.empty(0)
        for i in range(len(input_list)):
            input_tensor = torch.cat((input_tensor, input_list[i]), dim=0)
        
        label_tensor = torch.empty(0)     
        for i in range(len(label_list)):
            label_tensor = torch.cat((label_tensor, label_list[i]), dim=0)
        true_weights_tensor = torch.empty(0) 
        for i in range(len(true_weights_list)):
            true_weights_tensor = torch.cat((true_weights_tensor, true_weights_list[i]), dim=0)
            
        adv_dataset = TensorDataset(adv_input_tensor,input_tensor ,label_tensor, true_weights_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    
    
    def DBB_generate_adv_dataloader(model,test_dataloader,eps):
        neighbourhood_fn =  "8-grid"
        loss_fn = RegretLoss()
        comb_layer =  BlackboxDifflayer(lambda_val=args.lambda_val, neighbourhood_fn= neighbourhood_fn)
        
        adv_input_list = []
        input_list = []
        label_list = []
        true_weights_list = []
        for test_batch in test_dataloader:
            input, label, true_weights = test_batch
            input.requires_grad =  True
                
            output = model(input)
            weights = output.reshape(-1, output.shape[-1], output.shape[-1])
            
            ### For SPO, we need the true weights as we have to compute 2*\hat{c} - c
            shortest_path = comb_layer(weights)
            loss = loss_fn(shortest_path, label, true_weights)
            
            model.zero_grad()
            loss.backward()
            data_grad = input.grad.data
            adv_input = fgsm_attack(input, eps, data_grad)
            adv_input_list.append(adv_input.detach())
            input_list.append(input.detach())
            label_list.append(label)
            true_weights_list.append(true_weights)
            
        adv_input_tensor = torch.empty(0)    
        for i in range(len(adv_input_list)):
            adv_input_tensor = torch.cat((adv_input_tensor, adv_input_list[i]), dim=0)
        input_tensor = torch.empty(0)
        for i in range(len(input_list)):
            input_tensor = torch.cat((input_tensor, input_list[i]), dim=0)
            
        
        label_tensor = torch.empty(0)     
        for i in range(len(label_list)):
            label_tensor = torch.cat((label_tensor, label_list[i]), dim=0)
        true_weights_tensor = torch.empty(0) 
        for i in range(len(true_weights_list)):
            true_weights_tensor = torch.cat((true_weights_tensor, true_weights_list[i]), dim=0)
            
        adv_dataset = TensorDataset(adv_input_tensor,input_tensor ,label_tensor, true_weights_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    
    def FY_generate_adv_dataloader(model,test_dataloader,eps):
        neighbourhood_fn =  "8-grid"
        sigma = args.sigma
        num_samples = args.num_samples
        solver =   get_solver(neighbourhood_fn)
        fy_solver = lambda weights: shortest_pathsolution(solver, weights)
        criterion = fy.FenchelYoungLoss(fy_solver, num_samples= num_samples, sigma= sigma,maximize = False, batched= True)
        
        adv_input_list = []
        input_list = []
        label_list = []
        true_weights_list = []
        for test_batch in test_dataloader:
            input, label, true_weights = test_batch
            input.requires_grad =  True
                
            output = model(input)
            weights = output.reshape(-1, output.shape[-1], output.shape[-1])
            
        
            loss =  criterion(weights,label).mean()
            
            model.zero_grad()
            loss.backward()
            data_grad = input.grad.data
            adv_input = fgsm_attack(input, eps, data_grad)
            adv_input_list.append(adv_input.detach())
            input_list.append(input.detach())
            label_list.append(label)
            true_weights_list.append(true_weights)
            
        adv_input_tensor = torch.empty(0)    
        for i in range(len(adv_input_list)):
            adv_input_tensor = torch.cat((adv_input_tensor, adv_input_list[i]), dim=0)
        input_tensor = torch.empty(0)
        for i in range(len(input_list)):
            input_tensor = torch.cat((input_tensor, input_list[i]), dim=0)
        
        label_tensor = torch.empty(0)     
        for i in range(len(label_list)):
            label_tensor = torch.cat((label_tensor, label_list[i]), dim=0)
        true_weights_tensor = torch.empty(0) 
        for i in range(len(true_weights_list)):
            true_weights_tensor = torch.cat((true_weights_tensor, true_weights_list[i]), dim=0)
            
        adv_dataset = TensorDataset(adv_input_tensor,input_tensor ,label_tensor, true_weights_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    
    def IMLE_generate_adv_dataloader(model,test_dataloader,eps):
        neighbourhood_fn =  "8-grid"
        solver =   get_solver(neighbourhood_fn)

        target_distribution = TargetDistribution(alpha=1.0, beta=args.beta)
        noise_distribution = SumOfGammaNoiseDistribution(k= args.k, nb_iterations= args.nb_iterations)

        # @perturbations.perturbed(num_samples=num_samples, sigma=sigma, noise='gumbel',batched = False)
        imle_solver = imle(lambda weights: shortest_pathsolution(solver, weights),
        target_distribution=target_distribution,noise_distribution=noise_distribution,
        input_noise_temperature= args.temperature, target_noise_temperature= args.temperature, nb_samples= args.nb_samples)
        loss_fn = RegretLoss()
        
        
        adv_input_list = []
        input_list = []
        label_list = []
        true_weights_list = []
        for test_batch in test_dataloader:
            input, label, true_weights = test_batch
            input.requires_grad =  True
                
            output = model(input)
            weights = output.reshape(-1, output.shape[-1], output.shape[-1])

            shortest_path = imle_solver(weights)
            loss = loss_fn(shortest_path, label, true_weights)
            
            model.zero_grad()
            loss.backward()
            data_grad = input.grad.data
            adv_input = fgsm_attack(input, eps, data_grad)
            input_list.append(input.detach())
            adv_input_list.append(adv_input.detach())
            label_list.append(label)
            true_weights_list.append(true_weights)
            
        adv_input_tensor = torch.empty(0)    
        for i in range(len(adv_input_list)):
            adv_input_tensor = torch.cat((adv_input_tensor, adv_input_list[i]), dim=0)
        input_tensor = torch.empty(0)
        for i in range(len(input_list)):
            input_tensor = torch.cat((input_tensor, input_list[i]), dim=0)
        
        label_tensor = torch.empty(0)     
        for i in range(len(label_list)):
            label_tensor = torch.cat((label_tensor, label_list[i]), dim=0)
        true_weights_tensor = torch.empty(0) 
        for i in range(len(true_weights_list)):
            true_weights_tensor = torch.cat((true_weights_tensor, true_weights_list[i]), dim=0)
            
        adv_dataset = TensorDataset(adv_input_tensor,input_tensor ,label_tensor, true_weights_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    def baselineBCE_generate_adv_dataloader(model,test_dataloader,eps):
        criterion = nn.BCELoss()
        
        adv_input_list = []
        label_list = []
        input_list = []
        true_weights_list = []
        for test_batch in test_dataloader:
            input, label, true_weights = test_batch
            input.requires_grad =  True
                
            output = model(input)
            #TODO check if loss inputs have same dimensions
            flat_target = label.view(label.size()[0], -1)
            output = torch.sigmoid(output)
            print("lossBCE dimensions",output.size(),flat_target.size())
            loss = criterion(output, flat_target.to(torch.float32)).mean()
            
            model.zero_grad()
            loss.backward()
            data_grad = input.grad.data
            adv_input = fgsm_attack(input, eps, data_grad)
            adv_input_list.append(adv_input.detach())
            input_list.append(input.detach())
            label_list.append(label)
            true_weights_list.append(true_weights)
            
        adv_input_tensor = torch.empty(0)    
        for i in range(len(adv_input_list)):
            adv_input_tensor = torch.cat((adv_input_tensor, adv_input_list[i]), dim=0)
        input_tensor = torch.empty(0)
        for i in range(len(input_list)):
            input_tensor = torch.cat((input_tensor, input_list[i]), dim=0)
        
        label_tensor = torch.empty(0)     
        for i in range(len(label_list)):
            label_tensor = torch.cat((label_tensor, label_list[i]), dim=0)
        true_weights_tensor = torch.empty(0) 
        for i in range(len(true_weights_list)):
            true_weights_tensor = torch.cat((true_weights_tensor, true_weights_list[i]), dim=0)
            
        adv_dataset = TensorDataset(adv_input_tensor,input_tensor ,label_tensor, true_weights_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    
    def Listwise_generate_adv_dataloader(init_cache,model,test_dataloader,eps,growth=0.1):
        loss_fn = ListwiseLoss(tau=args.tau)
        neighbourhood_fn =  "8-grid"
        solver =   get_solver(neighbourhood_fn)
        
        init_cache_np = init_cache.detach().numpy()
        init_cache_np = np.unique(init_cache_np,axis=0)
        # torch has no unique function, so we have to do this
        cache = torch.from_numpy(init_cache_np).float()
        #growth = args.growth
        tau = args.tau
        #save_hyperparameters("lr","growth","tau")
        
        adv_input_list = []
        input_list = []
        label_list = []
        true_weights_list = []
        for test_batch in test_dataloader:
            input, label, true_weights = test_batch
            input.requires_grad =  True
                
            output = model(input)
            if (np.random.random(1)[0]<= growth) or len(cache)==0:
                cache = growcache(solver, cache, output)
            

            loss = loss_fn(output, true_weights, label , cache)
            model.zero_grad()
            loss.backward()
            data_grad = input.grad.data
            adv_input = fgsm_attack(input, eps, data_grad)
            adv_input_list.append(adv_input.detach())
            input_list.append(input.detach())
            label_list.append(label)
            true_weights_list.append(true_weights)
            
        adv_input_tensor = torch.empty(0)    
        for i in range(len(adv_input_list)):
            adv_input_tensor = torch.cat((adv_input_tensor, adv_input_list[i]), dim=0)
        input_tensor = torch.empty(0)
        for i in range(len(input_list)):
            input_tensor = torch.cat((input_tensor, input_list[i]), dim=0)
        
        label_tensor = torch.empty(0)     
        for i in range(len(label_list)):
            label_tensor = torch.cat((label_tensor, label_list[i]), dim=0)
        true_weights_tensor = torch.empty(0) 
        for i in range(len(true_weights_list)):
            true_weights_tensor = torch.cat((true_weights_tensor, true_weights_list[i]), dim=0)
            
        adv_dataset = TensorDataset(adv_input_tensor,input_tensor ,label_tensor, true_weights_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    def pairwise_generate_adv_dataloader(init_cache,model,test_dataloader,eps,growth=0.1):
        loss_fn = PairwiseLoss(tau=args.tau)
        neighbourhood_fn =  "8-grid"
        solver =   get_solver(neighbourhood_fn)
        
        init_cache_np = init_cache.detach().numpy()
        init_cache_np = np.unique(init_cache_np,axis=0)
        # torch has no unique function, so we have to do this
        cache = torch.from_numpy(init_cache_np).float()
        #growth = args.growth
        tau = args.tau
        #save_hyperparameters("lr","growth","tau")
        
        adv_input_list = []
        label_list = []
        input_list = []
        true_weights_list = []
        for test_batch in test_dataloader:
            input, label, true_weights = test_batch
            input.requires_grad =  True
                
            output = model(input)
            if (np.random.random(1)[0]<= growth) or len(cache)==0:
                cache = growcache(solver, cache, output)
            

            loss = loss_fn(output, true_weights, label , cache)
            model.zero_grad()
            loss.backward()
            data_grad = input.grad.data
            adv_input = fgsm_attack(input, eps, data_grad)
            adv_input_list.append(adv_input.detach())
            input_list.append(input.detach())
            label_list.append(label)
            true_weights_list.append(true_weights)
            
        adv_input_tensor = torch.empty(0)    
        for i in range(len(adv_input_list)):
            adv_input_tensor = torch.cat((adv_input_tensor, adv_input_list[i]), dim=0)
        input_tensor = torch.empty(0)
        for i in range(len(input_list)):
            input_tensor = torch.cat((input_tensor, input_list[i]), dim=0)
        
        label_tensor = torch.empty(0)     
        for i in range(len(label_list)):
            label_tensor = torch.cat((label_tensor, label_list[i]), dim=0)
        true_weights_tensor = torch.empty(0) 
        for i in range(len(true_weights_list)):
            true_weights_tensor = torch.cat((true_weights_tensor, true_weights_list[i]), dim=0)
            
        adv_dataset = TensorDataset(adv_input_tensor,input_tensor ,label_tensor, true_weights_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    
    def pairwiseDiff_generate_adv_dataloader(init_cache,model,test_dataloader,eps,growth=0.1):
        loss_fn = PairwisediffLoss()
        neighbourhood_fn =  "8-grid"
        solver =   get_solver(neighbourhood_fn)
        
        init_cache_np = init_cache.detach().numpy()
        init_cache_np = np.unique(init_cache_np,axis=0)
        # torch has no unique function, so we have to do this
        cache = torch.from_numpy(init_cache_np).float()
        #growth = args.growth
        tau = args.tau
        #save_hyperparameters("lr","growth","tau")
        
        adv_input_list = []
        input_list = []
        label_list = []
        true_weights_list = []
        for test_batch in test_dataloader:
            input, label, true_weights = test_batch
            input.requires_grad =  True
                
            output = model(input)
            if (np.random.random(1)[0]<= growth) or len(cache)==0:
                cache = growcache(solver, cache, output)
            

            loss = loss_fn(output, true_weights, label , cache)
            model.zero_grad()
            loss.backward()
            data_grad = input.grad.data
            adv_input = fgsm_attack(input, eps, data_grad)
            adv_input_list.append(adv_input.detach())
            input_list.append(input.detach())
            label_list.append(label)
            true_weights_list.append(true_weights)
            
        adv_input_tensor = torch.empty(0)    
        for i in range(len(adv_input_list)):
            adv_input_tensor = torch.cat((adv_input_tensor, adv_input_list[i]), dim=0)
        input_tensor = torch.empty(0)
        for i in range(len(input_list)):
            input_tensor = torch.cat((input_tensor, input_list[i]), dim=0)
        
        label_tensor = torch.empty(0)     
        for i in range(len(label_list)):
            label_tensor = torch.cat((label_tensor, label_list[i]), dim=0)
        true_weights_tensor = torch.empty(0) 
        for i in range(len(true_weights_list)):
            true_weights_tensor = torch.cat((true_weights_tensor, true_weights_list[i]), dim=0)
            
        adv_dataset = TensorDataset(adv_input_tensor,input_tensor ,label_tensor, true_weights_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    
    def MAP_generate_adv_dataloader(init_cache,model,test_dataloader,eps,growth=0.1):
        loss_fn = MAP()
        neighbourhood_fn =  "8-grid"
        solver =   get_solver(neighbourhood_fn)
        
        init_cache_np = init_cache.detach().numpy()
        init_cache_np = np.unique(init_cache_np,axis=0)
        # torch has no unique function, so we have to do this
        cache = torch.from_numpy(init_cache_np).float()
        #growth = args.growth
        tau = args.tau
        #save_hyperparameters("lr","growth","tau")
        
        adv_input_list = []
        input_list = []
        label_list = []
        true_weights_list = []
        for test_batch in test_dataloader:
            input, label, true_weights = test_batch
            input.requires_grad =  True
                
            output = model(input)
            if (np.random.random(1)[0]<= growth) or len(cache)==0:
                cache = growcache(solver, cache, output)
            

            loss = loss_fn(output, true_weights, label , cache)
            model.zero_grad()
            loss.backward()
            data_grad = input.grad.data
            adv_input = fgsm_attack(input, eps, data_grad)
            adv_input_list.append(adv_input.detach())
            input_list.append(input.detach())
            label_list.append(label)
            true_weights_list.append(true_weights)
            
        adv_input_tensor = torch.empty(0)    
        for i in range(len(adv_input_list)):
            adv_input_tensor = torch.cat((adv_input_tensor, adv_input_list[i]), dim=0)
        input_tensor = torch.empty(0)
        for i in range(len(input_list)):
            input_tensor = torch.cat((input_tensor, input_list[i]), dim=0)
        
        label_tensor = torch.empty(0)     
        for i in range(len(label_list)):
            label_tensor = torch.cat((label_tensor, label_list[i]), dim=0)
        true_weights_tensor = torch.empty(0) 
        for i in range(len(true_weights_list)):
            true_weights_tensor = torch.cat((true_weights_tensor ,true_weights_list[i]), dim=0)
        
        adv_dataset = TensorDataset(adv_input_tensor,input_tensor ,label_tensor, true_weights_tensor) 
        #adv_dataset = TensorDataset(adv_input_tensor, label_tensor, true_weights_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
    
    
            
            

            

       
    for seed in range(10): #break
    
        #if seed == 1 or seed == 9:
        #   continue
        
    #argument_dict['seed'] = seed

        ################## Define the outputfile
        outputfile = "Rslt/{}{}{}_index{}.csv".format(modelname,args.loss, img_size, index)
        regretfile = "Rslt/{}{}{}Regret_index{}.csv".format(modelname,args.loss, img_size, index)
        ckpt_dir =  "ckpt_dir/{}{}{}_index{}/".format(modelname, args.loss, img_size, index)
        log_dir = "lightning_logs/{}{}{}_index{}/".format(modelname, args.loss, img_size, index)
        learning_curve_datafile = "LearningCurve/{}{}{}_".format(modelname, args.loss, img_size)+"_".join( ["{}_{}".format(k,v) for k,v  in explicit.items() ]  )+".csv"

        for directory in [os.path.dirname(outputfile), os.path.dirname(regretfile), ckpt_dir, log_dir, os.path.dirname(learning_curve_datafile)]:
                if not os.path.exists(directory):
                    os.makedirs(directory)

        shutil.rmtree(log_dir,ignore_errors=True)


        ###################### Training Module   ######################

        seed_all(seed)

        g = torch.Generator()
        g.manual_seed(seed)

        data = WarcraftDataModule(data_dir="data/warcraft_shortest_path/{}".format(img_size), 
        batch_size= argument_dict['batch_size'], generator=g)
        metadata = data.metadata


        shutil.rmtree(ckpt_dir,ignore_errors=True)
        checkpoint_callback = ModelCheckpoint(
                monitor= "val_regret",
                dirpath=ckpt_dir, 
                filename="model-{epoch:02d}-{val_loss:.8f}",
                mode="min")

        tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)
        trainer = pl.Trainer(max_epochs= argument_dict['max_epochs'],
        min_epochs=1,logger=tb_logger, callbacks=[checkpoint_callback])
        if modelname=="CachingPO":
            cache = return_trainlabel(data_dir="data/warcraft_shortest_path/{}".format(img_size))
            #######YF
            test_cache = return_testlabel(data_dir="data/warcraft_shortest_path/{}".format(img_size))
            #########YF
            model = modelcls(metadata=metadata,init_cache=cache,seed=seed, **argument_dict)
        else:
            model = modelcls(metadata=metadata,seed=seed,**argument_dict)

        #######YF
        #SPOgenerate_adv_dataloader(model,data.test_dataloader(),eps)
        ##############
        trainer.fit(model, datamodule=data)

        best_model_path = checkpoint_callback.best_model_path
        if modelname=="CachingPO":
            model = modelcls.load_from_checkpoint(best_model_path,metadata=metadata,init_cache=cache,seed=seed ,**argument_dict)
        else:
            model = modelcls.load_from_checkpoint(best_model_path,metadata=metadata,seed=seed ,**argument_dict)


        regret_list = trainer.predict(model, data.test_dataloader())

        df = pd.DataFrame({"regret":regret_list[0].tolist()})
        df.index.name='instance'
        for k,v in explicit.items():
            df[k] = v
        df['seed'] = seed
        with open(regretfile, 'a') as f:
            df.to_csv(f, header=f.tell()==0)
            
        
        eps_list = [0.01, 0.1, 0.15]
        for eps in eps_list:
            ##### SummaryWrite ######################
            validresult = trainer.validate(model,datamodule=data)
            testresult = trainer.test(model, datamodule=data)


            ################YF
            if modelname == "CachingPO" and args.loss == "MAP":
                adv_dataloader = MAP_generate_adv_dataloader(test_cache,model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
                #adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}


                MSEadv_dataloader = MSE_generate_adv_dataloader(model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            
            
            if modelname == "CachingPO" and args.loss == "pairwise_diff":
                adv_dataloader = pairwiseDiff_generate_adv_dataloader(test_cache,model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
                #adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}


                MSEadv_dataloader = MSE_generate_adv_dataloader(model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            
            if modelname == "CachingPO" and args.loss == "pairwise":
                adv_dataloader = pairwise_generate_adv_dataloader(test_cache,model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
                #adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}


                MSEadv_dataloader = MSE_generate_adv_dataloader(model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            
            
            if modelname == "CachingPO" and args.loss == "listwise":
                adv_dataloader = Listwise_generate_adv_dataloader(test_cache,model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
            # adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}


                MSEadv_dataloader = MSE_generate_adv_dataloader(model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            
            if modelname == "baseline":
                adv_dataloader = baselineBCE_generate_adv_dataloader(model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
            # adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}


                MSEadv_dataloader = MSE_generate_adv_dataloader(model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            
            if modelname == "IMLE":
                adv_dataloader = IMLE_generate_adv_dataloader(model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
            # adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}


                MSEadv_dataloader = MSE_generate_adv_dataloader(model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            
            if modelname == "FenchelYoung":
                adv_dataloader = FY_generate_adv_dataloader(model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
                #adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}


                MSEadv_dataloader = MSE_generate_adv_dataloader(model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            
            
            if modelname == "DBB":
                adv_dataloader = DBB_generate_adv_dataloader(model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
            # adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}


                MSEadv_dataloader = MSE_generate_adv_dataloader(model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
            
            if modelname == "SPO":
                adv_dataloader = SPO_generate_adv_dataloader(model,data.test_dataloader(),eps)
                adv_testresult = trainer.test(model, adv_dataloader)
                #adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}


                MSEadv_dataloader = MSE_generate_adv_dataloader(model,data.test_dataloader(),eps)
                MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
                MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}


            ###############YF

            df = pd.DataFrame({**MSEadv_testresult[0],**adv_testresult[0], **testresult[0], **validresult[0]},index=[0])
            for k,v in explicit.items():
                df[k] = v
            df['seed'] = seed
            df['epsilon'] = eps
            with open(outputfile, 'a') as f:
                df.to_csv(f, header=f.tell()==0)

        ##### Save Learning Curve Data ######################
        parent_dir=   log_dir+"lightning_logs/"
        version_dirs = [os.path.join(parent_dir,v) for v in os.listdir(parent_dir)]

        walltimes = []
        steps = []
        regrets= []
        hammings=[]
        mses = []
        for logs in version_dirs:
            event_accumulator = EventAccumulator(logs)
            event_accumulator.Reload()

            events = event_accumulator.Scalars("val_hammingloss")
            walltimes.extend( [x.wall_time for x in events])
            steps.extend([x.step for x in events])
            hammings.extend([x.value for x in events])
            events = event_accumulator.Scalars("val_regret")
            regrets.extend([x.value for x in events])
            events = event_accumulator.Scalars("val_mse")
            mses.extend([x.value for x in events])

        df = pd.DataFrame({"step": steps,'wall_time':walltimes,  "val_hammingloss": hammings, "val_regret": regrets,
        "val_mse": mses })
        df['model'] = modelname + args.loss
        df['seed'] = seed
        #df['epsilon'] = eps
        df.to_csv(learning_curve_datafile,index=False)
