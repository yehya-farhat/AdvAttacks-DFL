from Trainer.PO_modelsSP import *
import pandas as pd
import shutil
from pytorch_lightning.callbacks import ModelCheckpoint 
import random
from pytorch_lightning import loggers as pl_loggers
from Trainer.data_utils import datawrapper, ShortestPathDataModule
torch.use_deterministic_algorithms(True)
import argparse
from argparse import Namespace
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from distutils.util import strtobool
from Trainer.optimizer_module import gurobi_portfolio_solver
from torch.utils.data import DataLoader, TensorDataset
from Trainer.CacheLosses import *

net_layers = [nn.BatchNorm1d(5),nn.Linear(5,50)]
batchnorm_net = nn.Sequential(*net_layers)
nonorm_net = nn.Linear(5,50)
class normalized_network(nn.Module):
    def __init__(self):
        super().__init__()
        # self.batchnorm = nn.BatchNorm1d(5)
        self.fc = nn.Linear(5,50)

    def forward(self, x):
        # x = self.batchnorm(x)
        x = self.fc(x)
        return F.normalize(x, p=1,dim = 1)
l1norm_net =  normalized_network()
network_dict = {"nonorm":nonorm_net, "batchnorm":batchnorm_net, "l1norm":l1norm_net}

parser = argparse.ArgumentParser()

parser.add_argument("--N", type=int, help="Dataset size", default= 1000, required= True)
parser.add_argument("--noise", type= int, help="noise halfwidth paraemter", default= 1, required= True)
parser.add_argument("--deg", type=int, help="degree of misspecifaction", default= 1000, required= True)

parser.add_argument("--model", type=str, help="name of the model", default= "", required= True)
parser.add_argument("--loss", type= str, help="loss", default= "", required=False)
parser.add_argument("--net", type=str, help="Type of Model Archietcture, one of: nonorm, batchnorm,l1norm", default= "l1norm", required= False)

parser.add_argument("--lr", type=float, help="learning rate", default= 1e-3, required=False)
parser.add_argument("--batch_size", type=int, help="batch size", default= 128, required=False)
parser.add_argument("--max_epochs", type=int, help="maximum number of epochs", default= 30, required=False)
parser.add_argument("--l1_weight", type=float, help="Weight of L1 regularization", default= 1e-5, required=False)


parser.add_argument("--lambda_val", type=float, help="interpolaton parameter blackbox", default= 1., required=False)
parser.add_argument("--sigma", type=float, help="DPO FY noise parameter", default= 1., required=False)
parser.add_argument("--num_samples", type=int, help="number of samples FY", default= 1, required=False)

parser.add_argument("--temperature", type=float, help="input and target noise temperature parameter", default= 1., required=False)
parser.add_argument("--nb_iterations", type=int, help="number of iterations", default= 10, required=False)
parser.add_argument("--k", type=int, help="parameter k", default= 10, required=False)
parser.add_argument("--nb_samples", type=int, help="Number of samples paprameter", default= 10, required=False)
parser.add_argument("--beta", type=float, help="parameter lambda of IMLE", default= 10., required=False)
#parser.add_argument("--eps", type=float, help="adv pertubation", default= 0.0, required=False)


parser.add_argument("--mu", type=float, help="Regularization parameter DCOL & QPTL", default= 10., required=False)
parser.add_argument("--regularizer", type=str, help="Types of Regularization", default= 'quadratic', required=False)
parser.add_argument("--thr", type=float, help="threshold parameter", default= 1e-6)
parser.add_argument("--damping", type=float, help="damping parameter", default= 1e-8)
parser.add_argument("--diffKKT",  action='store_true', help="Whether KKT or HSD ",  required=False)

parser.add_argument("--tau", type=float, help="parameter of rankwise losses", default= 1e-8)
parser.add_argument("--growth", type=float, help="growth parameter of rankwise losses", default= 1e-8)

parser.add_argument('--scheduler', dest='scheduler',  type=lambda x: bool(strtobool(x)))
args = parser.parse_args()
class _Sentinel:
    pass
sentinel = _Sentinel()

argument_dict = vars(args)
get_class = lambda x: globals()[x]
modelcls = get_class(argument_dict['model'])
modelname = argument_dict.pop('model')
network_name = argument_dict.pop('net')



N = argument_dict.pop('N')
noise = argument_dict.pop('noise')
deg = argument_dict.pop('deg')



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
    
########### YF adv attacks
def fgsm_attack(x, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        perturbed_x = x + epsilon*sign_data_grad
        return perturbed_x
    
    
def generate_adv_dataloader_MSE(l1_weight,model,test_dataloader,eps):
        lossfn = nn.MSELoss()
        #comblayer = cvx_knapsack_solver(weights,capacity,n_items,mu=args.mu)
        adv_x_list = []
        x_list = []
        y_list = []
        sol_list = []
        for test_batch in test_dataloader:
            # Send the data and label to the device
            
            
            x,y,sol = test_batch
            x.requires_grad =  True
   
            
            y_hat =  model(x).squeeze()
            
           # l1penalty = sum([(param.abs()).sum() for param in model.net.parameters()])
            
            loss = lossfn(y_hat,y)# + l1penalty * l1_weight
           
            model.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            adv_x = fgsm_attack(x, eps, data_grad)
            adv_x_list.append(adv_x.detach())
            x_list.append(x.detach())
            y_list.append(y)
            sol_list.append(sol)
           
        adv_x_tensor = torch.empty(0)    
        for i in range(len(adv_x_list)):
            adv_x_tensor = torch.cat((adv_x_tensor, adv_x_list[i]), dim=0)
            
        x_tensor = torch.empty(0)
        for i in range(len(x_list)):
            x_tensor = torch.cat((x_tensor, x_list[i]), dim=0)
        
        y_tensor = torch.empty(0)     
        for i in range(len(y_list)):
            y_tensor = torch.cat((y_tensor, y_list[i]), dim=0)
        sol_tensor = torch.empty(0) 
        for i in range(len(sol_list)):
            sol_tensor = torch.cat((sol_tensor, sol_list[i]), dim=0)

        # Create a new DataLoader with adv_x, y, and sol
        adv_dataset = TensorDataset(adv_x_tensor,x_tensor ,y_tensor, sol_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
def DCOL_generate_adv_dataloader(cov, gamma, l1_weight,model,test_dataloader,eps):
        layer = cvxsolver( cov=cov, gamma=gamma,  mu=args.mu, regularizer=args.regularizer)
        
        adv_x_list = []
        x_list = []
        y_list = []
        sol_list = []

        for test_batch in test_dataloader:
            # Send the data and label to the device
            x,y,sol = test_batch
            x.requires_grad =  True    
        
            y_hat =  model(x).squeeze()
            loss = 0
           # l1penalty = sum([(param.abs()).sum() for param in model.net.parameters()])

            sol_hat = layer.solution(y_hat)
            loss =  ((sol_hat - sol)*y).sum(-1).mean() #+ l1penalty * l1_weight
            
            
            
            model.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            adv_x = fgsm_attack(x, eps, data_grad)
            adv_x_list.append(adv_x.detach())
            x_list.append(x.detach())
            y_list.append(y)
            sol_list.append(sol)
            
        adv_x_tensor = torch.empty(0)    
        for i in range(len(adv_x_list)):
            adv_x_tensor = torch.cat((adv_x_tensor, adv_x_list[i]), dim=0)
            
        x_tensor = torch.empty(0)
        for i in range(len(x_list)):
            x_tensor = torch.cat((x_tensor, x_list[i]), dim=0)
        
        y_tensor = torch.empty(0)     
        for i in range(len(y_list)):
            y_tensor = torch.cat((y_tensor, y_list[i]), dim=0)
        sol_tensor = torch.empty(0) 
        for i in range(len(sol_list)):
            sol_tensor = torch.cat((sol_tensor, sol_list[i]), dim=0)

        # Create a new DataLoader with adv_x, y, and sol
        adv_dataset = TensorDataset(adv_x_tensor,x_tensor, y_tensor, sol_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
def SPO_generate_adv_dataloader(exact_solver, l1_weight,model,test_dataloader,eps):
        loss_fn =  SPOlayer(exact_solver)
        
        adv_x_list = []
        x_list = []
        y_list = []
        sol_list = []

        for test_batch in test_dataloader:
            # Send the data and label to the device
            x,y,sol = test_batch
            x.requires_grad =  True    
        
            y_hat =  model(x).squeeze()
            #loss = 0
            #l1penalty = sum([(param.abs()).sum() for param in model.net.parameters()])
            # for ii in range(len(y)):
            #     loss += self.loss_fn(y_hat[ii],y[ii], sol[ii])
            loss = loss_fn(y_hat,y, sol)#/len(y) + l1penalty * self.l1_weight
            
            model.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            adv_x = fgsm_attack(x, eps, data_grad)
            adv_x_list.append(adv_x.detach())
            x_list.append(x.detach())
            y_list.append(y)
            sol_list.append(sol)
            
        adv_x_tensor = torch.empty(0)    
        for i in range(len(adv_x_list)):
            adv_x_tensor = torch.cat((adv_x_tensor, adv_x_list[i]), dim=0)
            
        x_tensor = torch.empty(0)
        for i in range(len(x_list)):
            x_tensor = torch.cat((x_tensor, x_list[i]), dim=0)
        
        y_tensor = torch.empty(0)     
        for i in range(len(y_list)):
            y_tensor = torch.cat((y_tensor, y_list[i]), dim=0)
        sol_tensor = torch.empty(0) 
        for i in range(len(sol_list)):
            sol_tensor = torch.cat((sol_tensor, sol_list[i]), dim=0)

        # Create a new DataLoader with adv_x, y, and sol
        adv_dataset = TensorDataset(adv_x_tensor,x_tensor ,y_tensor, sol_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
def DBB_generate_adv_dataloader(exact_solver, l1_weight,model,test_dataloader,eps):
        lambda_val = args.lambda_val
        layer = DBBlayer(exact_solver,lambda_val)
        #self.save_hyperparameters("lr","lambda_val")
        
        adv_x_list = []
        x_list = []
        y_list = []
        sol_list = []

        for test_batch in test_dataloader:
            # Send the data and label to the device
            x,y,sol = test_batch
            x.requires_grad =  True    
        
            y_hat =  model(x).squeeze()
            sol_hat = layer(y_hat, y, sol)
            #l1penalty = sum([(param.abs()).sum() for param in model.net.parameters()])

            loss =  ((sol_hat - sol)*y).sum(-1).mean()# + l1penalty * l1_weight
            
            model.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            adv_x = fgsm_attack(x, eps, data_grad)
            adv_x_list.append(adv_x.detach())
            x_list.append(x.detach())
            y_list.append(y)
            sol_list.append(sol)
            
        adv_x_tensor = torch.empty(0)    
        for i in range(len(adv_x_list)):
            adv_x_tensor = torch.cat((adv_x_tensor, adv_x_list[i]), dim=0)
            
        x_tensor = torch.empty(0)
        for i in range(len(x_list)):
            x_tensor = torch.cat((x_tensor, x_list[i]), dim=0)
        
        y_tensor = torch.empty(0)     
        for i in range(len(y_list)):
            y_tensor = torch.cat((y_tensor, y_list[i]), dim=0)
        sol_tensor = torch.empty(0) 
        for i in range(len(sol_list)):
            sol_tensor = torch.cat((sol_tensor, sol_list[i]), dim=0)
        # Create a new DataLoader with adv_x, y, and sol
        adv_dataset = TensorDataset(adv_x_tensor,x_tensor ,y_tensor, sol_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
def IMLE_generate_adv_dataloader(exact_solver, l1_weight,model,test_dataloader,eps):
    solver = exact_solver
    k = args.k
    nb_iterations = args.nb_iterations
    nb_samples = args.nb_samples
    
    target_distribution = TargetDistribution(alpha=1.0, beta= args.beta)
    noise_distribution = SumOfGammaNoiseDistribution(k= k, nb_iterations=nb_iterations)

    imle_solver = lambda y_: solver.solution_fromtorch(-y_)

    imle_layer = imle(imle_solver,target_distribution=target_distribution,
    noise_distribution=noise_distribution, input_noise_temperature= args.temperature, 
    target_noise_temperature= args.temperature,nb_samples=nb_samples)

    adv_x_list = []
    x_list = []
    y_list = []
    sol_list = []

    for test_batch in test_dataloader:
        # Send the data and label to the device
        x,y,sol = test_batch
        x.requires_grad =  True    
    
        y_hat =  model(x).squeeze()
        #l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        
        sol_hat = imle_layer(-y_hat)
        loss = ((sol_hat - sol)*y).sum(-1).mean()
        #training_loss= loss  + l1penalty * l1_weight
        
        model.zero_grad()
        loss.backward()
        data_grad = x.grad.data
        adv_x = fgsm_attack(x, eps, data_grad)
        adv_x_list.append(adv_x.detach())
        x_list.append(x.detach())
        y_list.append(y)
        sol_list.append(sol)
        
    adv_x_tensor = torch.empty(0)    
    for i in range(len(adv_x_list)):
        adv_x_tensor = torch.cat((adv_x_tensor, adv_x_list[i]), dim=0)
        
    x_tensor = torch.empty(0)
    for i in range(len(x_list)):
        x_tensor = torch.cat((x_tensor, x_list[i]), dim=0)
    
    y_tensor = torch.empty(0)     
    for i in range(len(y_list)):
        y_tensor = torch.cat((y_tensor, y_list[i]), dim=0)
    sol_tensor = torch.empty(0) 
    for i in range(len(sol_list)):
        sol_tensor = torch.cat((sol_tensor, sol_list[i]), dim=0)

    # Create a new DataLoader with adv_x, y, and sol
    adv_dataset = TensorDataset(adv_x_tensor,x_tensor ,y_tensor, sol_tensor)
    new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
    return new_dataloader


def FY_generate_adv_dataloader(exact_solver, l1_weight,model,test_dataloader,eps):
        solver = exact_solver
        num_samples = args.num_samples
        sigma = args.sigma
        #save_hyperparameters("lr")
        fy_solver = lambda y_: exact_solver.solution_fromtorch(y_)
        
        adv_x_list = []
        x_list = []
        y_list = []
        sol_list = []

        for test_batch in test_dataloader:
            # Send the data and label to the device
            x,y,sol = test_batch
            x.requires_grad =  True    
        
            y_hat =  model(x).squeeze()
            loss = 0
     

            criterion = fy.FenchelYoungLoss(fy_solver, num_samples= num_samples, sigma= sigma, maximize = False,
            batched=True)
            #l1penalty = sum([(param.abs()).sum() for param in model.net.parameters()])
            loss = criterion(y_hat, sol).mean()
    
            model.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            adv_x = fgsm_attack(x, eps, data_grad)
            adv_x_list.append(adv_x.detach())
            x_list.append(x.detach())
            y_list.append(y)
            sol_list.append(sol)
            
        adv_x_tensor = torch.empty(0)    
        for i in range(len(adv_x_list)):
            adv_x_tensor = torch.cat((adv_x_tensor, adv_x_list[i]), dim=0)
            
        x_tensor = torch.empty(0)
        for i in range(len(x_list)):
            x_tensor = torch.cat((x_tensor, x_list[i]), dim=0)
        
        y_tensor = torch.empty(0)     
        for i in range(len(y_list)):
            y_tensor = torch.cat((y_tensor, y_list[i]), dim=0)
        sol_tensor = torch.empty(0) 
        for i in range(len(sol_list)):
            sol_tensor = torch.cat((sol_tensor, sol_list[i]), dim=0)

        # Create a new DataLoader with adv_x, y, and sol
        adv_dataset = TensorDataset(adv_x_tensor,x_tensor ,y_tensor, sol_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader
    
def CachingPO_generate_adv_dataloader(init_cache,exact_solver, l1_weight,model,test_dataloader,eps):
        
        if args.loss =="pairwise":
            loss_fn = PairwiseLoss(margin=args.tau)
        elif args.loss == "pairwise_diff":
            loss_fn = PairwisediffLoss()
        elif args.loss == "listwise":
            loss_fn = ListwiseLoss(temperature=args.tau)
        elif args.loss == 'MAP':
            loss_fn = MAP()
        else:
            raise Exception("Invalid Loss Provided")
        
        init_cache_np = init_cache.detach().numpy()
        init_cache_np = np.unique(init_cache_np,axis=0)
        # torch has no unique function, so we have to do this
        cache = torch.from_numpy(init_cache_np).float()
        growth = args.growth
        #tau = args.tau
        #self.save_hyperparameters("lr","growth","tau")
        
        adv_x_list = []
        x_list = []
        y_list = []
        sol_list = []

        for test_batch in test_dataloader:
            # Send the data and label to the device
            x,y,sol = test_batch
            x.requires_grad =  True    
        
            y_hat =  model(x).squeeze()
            
            if (np.random.random(1)[0]<= growth) or len(cache)==0:
                cache = growcache(exact_solver, cache, y_hat)

  
            loss = loss_fn(y_hat,y,sol,cache)
           # l1penalty = sum([(param.abs()).sum() for param in model.net.parameters()])
            #loss=  loss/len(y) # + l1penalty * l1_weight
            
            
    
            model.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            adv_x = fgsm_attack(x, eps, data_grad)
            adv_x_list.append(adv_x.detach())
            x_list.append(x.detach())
            y_list.append(y)
            sol_list.append(sol)
            
        adv_x_tensor = torch.empty(0)    
        for i in range(len(adv_x_list)):
            adv_x_tensor = torch.cat((adv_x_tensor, adv_x_list[i]), dim=0)
            
        x_tensor = torch.empty(0)
        for i in range(len(x_list)):
            x_tensor = torch.cat((x_tensor, x_list[i]), dim=0)
        
        y_tensor = torch.empty(0)     
        for i in range(len(y_list)):
            y_tensor = torch.cat((y_tensor, y_list[i]), dim=0)
        sol_tensor = torch.empty(0) 
        for i in range(len(sol_list)):
            sol_tensor = torch.cat((sol_tensor, sol_list[i]), dim=0)

        # Create a new DataLoader with adv_x, y, and sol
        adv_dataset = TensorDataset(adv_x_tensor,x_tensor ,y_tensor, sol_tensor)
        new_dataloader = DataLoader(adv_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        return new_dataloader

################## Define the outputfile
outputfile = "Rslt/{}{}.csv".format(modelname,args.loss)
regretfile = "Rslt/{}{}Regret.csv".format(modelname,args.loss)
ckpt_dir =  "ckpt_dir/{}{}/".format(modelname, args.loss)
log_dir = "lightning_logs/{}{}/".format(modelname, args.loss)
learning_curve_datafile =    "LearningCurve/{}_".format(modelname)+"_".join( ["{}_{}".format(k,v) for k,v  in explicit.items()] )+".csv" 


################## DataReading
Train_dfx= pd.read_csv("SyntheticPortfolioData/TraindataX_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None)
Train_dfy= pd.read_csv("SyntheticPortfolioData/Traindatay_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None,)
x_train =  Train_dfx.T.values.astype(np.float32)
y_train = Train_dfy.T.values.astype(np.float32)

Validation_dfx= pd.read_csv("SyntheticPortfolioData/ValidationdataX_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None)
Validation_dfy= pd.read_csv("SyntheticPortfolioData/Validationdatay_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None,)
x_valid =  Validation_dfx.T.values.astype(np.float32)
y_valid = Validation_dfy.T.values.astype(np.float32)

Test_dfx= pd.read_csv("SyntheticPortfolioData/TestdataX_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None)
Test_dfy= pd.read_csv("SyntheticPortfolioData/Testdatay_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None,)
x_test =  Test_dfx.T.values.astype(np.float32)
y_test = Test_dfy.T.values.astype(np.float32)
data =  np.load('SyntheticPortfolioData/GammaSigma_N_{}_noise_{}_deg_{}.npz'.format(N,noise,deg))
cov = data['sigma']
gamma = data['gamma']
portfolio_solver = gurobi_portfolio_solver(cov= cov, gamma=gamma)


train_df =  datawrapper( x_train,y_train, solver=portfolio_solver )
valid_df =  datawrapper( x_valid,y_valid, solver=portfolio_solver)
test_df =  datawrapper( x_test,y_test, solver=portfolio_solver)
shutil.rmtree(log_dir,ignore_errors=True)


for seed in range(10):
    seed_all(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    data = ShortestPathDataModule(train_df, valid_df, test_df, generator=g, num_workers=8)
    net = network_dict[network_name]


    shutil.rmtree(ckpt_dir,ignore_errors=True)
    checkpoint_callback = ModelCheckpoint(
        # monitor="val_regret",mode="min",
        dirpath=ckpt_dir, 
        filename="model-{epoch:02d}-{val_loss:.2f}",
    )
    
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)
    trainer = pl.Trainer(max_epochs= argument_dict['max_epochs'], callbacks=[checkpoint_callback],  min_epochs=5, logger=tb_logger)
    if modelname=="CachingPO":
        init_cache = batch_solve(portfolio_solver, torch.from_numpy(y_train),relaxation =False)
        #####YF
        test_cache = batch_solve(portfolio_solver, torch.from_numpy(y_test),relaxation =False)
        #####YF
        model = modelcls(exact_solver = portfolio_solver, cov=cov, gamma=gamma,  net= net,seed=seed, init_cache=init_cache, **argument_dict)
    else:
        model = modelcls(exact_solver = portfolio_solver, cov=cov, gamma=gamma, net= net ,seed=seed, **argument_dict)


    validresult = trainer.validate(model,datamodule=data)
    trainer.fit(model, datamodule=data)
    
    best_model_path = checkpoint_callback.best_model_path
    if modelname=="CachingPO":
        init_cache = batch_solve(portfolio_solver, torch.from_numpy(y_train),relaxation =False)
        model = modelcls.load_from_checkpoint(best_model_path,  exact_solver = portfolio_solver, cov=cov, gamma=gamma,  seed=seed, net= net, init_cache=init_cache, **argument_dict)
    else:
        model = modelcls.load_from_checkpoint(best_model_path,   exact_solver = portfolio_solver, cov=cov, gamma=gamma,  net= net,seed=seed, **argument_dict)


    y_pred = model(torch.from_numpy(x_test).float()).squeeze()
    sol_test =  batch_solve(portfolio_solver, torch.from_numpy(y_test).float())
    regret_list_data = regret_list(portfolio_solver, y_pred, torch.from_numpy(y_test).float(), sol_test)

    df = pd.DataFrame({"regret":regret_list_data})
    df.index.name='instance'
    df['seed'] =seed
    for k,v in explicit.items():
        df[k] = v
    with open(regretfile, 'a') as f:
        df.to_csv(f, header=f.tell()==0)

    eps_list = [0.01, 0.1, 0.15]
    for eps in eps_list:
            
        ##### Summary
        validresult = trainer.validate(model,datamodule=data)
        testresult = trainer.test(model, datamodule=data)
        #################YF
        
        
        if modelname == "baseline":
            adv_dataloader = DCOL_generate_adv_dataloader(cov, gamma,args.l1_weight,model,data.test_dataloader(),eps)
            adv_testresult = trainer.test(model, adv_dataloader)
            #adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

        
            MSEadv_dataloader = generate_adv_dataloader_MSE(args.l1_weight, model,data.test_dataloader(),eps)
            MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
            MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
        if modelname == "CachingPO":
            adv_dataloader = CachingPO_generate_adv_dataloader(test_cache,portfolio_solver,args.l1_weight,model,data.test_dataloader(),eps)
            adv_testresult = trainer.test(model, adv_dataloader)
            #adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

        
            MSEadv_dataloader = generate_adv_dataloader_MSE(args.l1_weight, model,data.test_dataloader(),eps)
            MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
            MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
        if modelname == "FenchelYoung":
            adv_dataloader = FY_generate_adv_dataloader(portfolio_solver,args.l1_weight,model,data.test_dataloader(),eps)
            adv_testresult = trainer.test(model, adv_dataloader)
            #adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

        
            MSEadv_dataloader = generate_adv_dataloader_MSE(args.l1_weight, model,data.test_dataloader(),eps)
            MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
            MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
        
            
        if modelname == "IMLE":
            adv_dataloader = IMLE_generate_adv_dataloader(portfolio_solver,args.l1_weight,model,data.test_dataloader(),eps)
            adv_testresult = trainer.test(model, adv_dataloader)
            #adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

        
            MSEadv_dataloader = generate_adv_dataloader_MSE(args.l1_weight, model,data.test_dataloader(),eps)
            MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
            MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
            
        if modelname == "DBB":
            adv_dataloader = DBB_generate_adv_dataloader(portfolio_solver,args.l1_weight,model,data.test_dataloader(),eps)
            adv_testresult = trainer.test(model, adv_dataloader)
            #adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

        
            MSEadv_dataloader = generate_adv_dataloader_MSE(args.l1_weight, model,data.test_dataloader(),eps)
            MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
            MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
        
        
        
        if modelname == "SPO":
            adv_dataloader = SPO_generate_adv_dataloader(portfolio_solver,args.l1_weight,model,data.test_dataloader(),eps)
            adv_testresult = trainer.test(model, adv_dataloader)
        # adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

        
            MSEadv_dataloader = generate_adv_dataloader_MSE(args.l1_weight, model,data.test_dataloader(),eps)
            MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
            MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
        
        if modelname == "DCOL":
            adv_dataloader = DCOL_generate_adv_dataloader(cov, gamma,args.l1_weight,model,data.test_dataloader(),eps)
            adv_testresult = trainer.test(model, adv_dataloader)
        # adv_testresult[0] = {f'adv_{k}': v for k, v in adv_testresult[0].items()}

        
            MSEadv_dataloader = generate_adv_dataloader_MSE(args.l1_weight, model,data.test_dataloader(),eps)
            MSEadv_testresult = trainer.test(model, MSEadv_dataloader)
            MSEadv_testresult[0] = {f'MSE_{k}': v for k, v in MSEadv_testresult[0].items()}
        
        
        df = pd.DataFrame({**MSEadv_testresult[0],**adv_testresult[0],**testresult[0], **validresult[0]},index=[0])    
        df['seed'] =seed
        for k,v in explicit.items():
            df[k] = v
        df['eps'] = eps
        with open(outputfile, 'a') as f:
            df.to_csv(f, header=f.tell()==0)
###############################  Save  Learning Curve Data ########
import os
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

    events = event_accumulator.Scalars("val_abs_regret")
    walltimes.extend( [x.wall_time for x in events])
    steps.extend([x.step for x in events])
    regrets.extend([x.value for x in events])
    events = event_accumulator.Scalars("val_mse")
    mses.extend([x.value for x in events])

df = pd.DataFrame({"step": steps,'wall_time':walltimes,  "val_abs_regret": regrets,
"val_mse": mses })
for k,v in explicit.items():
    df[k] = v
with open( learning_curve_datafile, 'a') as f:
    df.to_csv(f, header=f.tell()==0)
