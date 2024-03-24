import wandb
import pickle 
import json 
from Utils import *
from Models import *
from Dataset import * 
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from proteinworkshop.features.factory import ProteinFeaturiser  

def train(args):

 
    # Load in dict with parameters for current run 
    with open(f'{args.run}parameters.json', 'r') as file:
        param = json.load(file)

    for key in param.keys():
        print(f'{key}: {param[key]}')

    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device('cuda')
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print(f'GPU: {torch.cuda.get_device_name(0)} will be used')

    # If GPU is not available
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device('cpu')

    # Initialize featuriser for data 
    featuriser = ProteinFeaturiser(
        representation = param['representation'],
        scalar_node_features = param['scalar_node_features'],
        vector_node_features=[],
        edge_types = param['edge_types'],
        scalar_edge_features = param['scalar_edge_features'],
        vector_edge_features=[]
        )

    # Load in model
    model = GraphEncDec(featuriser=featuriser,hidden_dim_GCN=param['hidden_dim_GCN'], decoder_type=param['decoder_type'], LSTM_hidden_dim=param['LSTM_hidden_dim'], dropout=param['dropout'], type=param['type'],LSTMnormalization=param['LSTMnormalization'] ,lstm_layers=param['lstm_layers'], encoder_concat_hidden=param['encoder_concat_hidden'], num_relation=len(param['edge_types']), input_dim=param['input_dim'])
    model.to(device)

    # Load in dict with information about each protein 
    with open(param['data_dir'] + 'data_partitions.pkl', 'rb') as file:
        data_dict = pickle.load(file)
    
    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr = param['lr'], weight_decay = param['weight_decay'])
    if param['criterion'] == 'CE':
        criterion = nn.CrossEntropyLoss(label_smoothing = param['label_smoothing'])

    # Datasplits for trianing and validation 
    data_splits = [[[0, 1, 2], [3]]]
    #            [[1, 2, 3], [4]],
    #            [[2, 3, 4], [0]],
    #            [[3, 4, 0], [1]],
    #            [[4, 0, 1], [2]]]
    """
    # smaller dataset with 5 fold
    data_splits = [[[0], [3]],
                [[1], [4]],
                [[2], [0]],
                [[3], [1]],
                [[4], [2]]]
    
       
    # small dataset with 1 fold  
    data_splits = [[[4], [3]]]
    """    
    # Start 5-fold CV
    k = 0
    for split in data_splits:
        k += 1
        
        if param['tracking']: 
            # Initialize weights and biases (tracking learning curves)
            wandb.login(key = param['API_key'])
            wandb.init(project = param['project_name'], config=param)
            run_number = args.run.rstrip('/').split('/')[-1].split('_')[-1]
            wandb.run.name = f'Run {run_number}' 
        
        # Initializing lists to store train+val acc+loss per epoch    
        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        # Training model from scratch 
        if param['training_point'] == 'begining':

            # Initializing weights
            lazy_initialization(param['data_dir'], data_dict, model, device)
            model.apply(custom_weights_init)

        # Continue training from last model saved
        elif param['training_point'] == 'last_epoch':
            continue_run_number = param['continue_run'].rstrip('/').split('/')[-1].split('_')[-1]
            print(f'Continue training for run: {continue_run_number}')
            model.load_state_dict(torch.load(param['continue_run'] + f'last_model.cv{k}.pt'))
            
        # Seperating proteins for train and validation 
        train_protein_list = cv_fold_split(data_dict, split[0])
        val_protein_list = cv_fold_split(data_dict, split[1])

        # Defining each dataset
        trainDset = GearNetDataset(root = param['data_dir'], protein_list = train_protein_list, data_dict = data_dict)
        valDset = GearNetDataset(root = param['data_dir'], protein_list = val_protein_list, data_dict = data_dict)  
        
        # Loading in datasets
        if param['weighted_sampling']:
            sampler_weights = weighted_sampling(data_dict, param, train_protein_list, device)
            sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights),replacement=True)
            train_loader = DataLoader(trainDset, batch_size = param['batch_size'], shuffle=False, num_workers = 0, sampler=sampler)
        else:
            train_loader = DataLoader(trainDset, batch_size = param['batch_size'], shuffle=False, num_workers = 0)
        val_loader = DataLoader(valDset, batch_size = param['batch_size'], shuffle = False, num_workers = 0)

        # Keeping track on best model per CV-fold
        best_acc = 0
        # Epoch counter
        c = 0 
        for epoch in range(param['num_epoch']):

            # chose which performance measures to use for tracking turing training ('residue' extends the training time)
            result_type =  param['result_type']
            
            # Training and validating one epoch
            if result_type == ['residue']:
                train_loss_per_epoch, train_per_residue_acc, _ = train_one_epoch(model, optimizer, criterion, train_loader, device, param['decoder_type'], result_type)
                val_loss_per_epoch, val_per_residue_acc, _ = val_one_epoch(model, criterion, val_loader, device, param['decoder_type'], result_type)
            elif result_type == ['class']: 
                train_loss_per_epoch, train_overlap_acc, train_residue_acc = train_one_epoch(model, optimizer, criterion, train_loader, device, param['decoder_type'], result_type)
                val_loss_per_epoch, val_overlap_acc, val_residue_acc = val_one_epoch(model, criterion, val_loader, device, param['decoder_type'], result_type)
            else:
                train_loss_per_epoch, train_overlap_acc, train_residue_acc, train_per_residue_acc, _ = train_one_epoch(model,optimizer, criterion, train_loader, device, param['decoder_type'], result_type)
                val_loss_per_epoch, val_overlap_acc, val_residue_acc, val_per_residue_acc, _ = val_one_epoch(model, criterion, val_loader, device, param['decoder_type'], result_type)

            #train_loss_per_epoch,train_overlap_acc,train_residue_acc = train_one_epoch(model,optimizer, criterion, train_loader, device, param['decoder_type'], result_type)  
            #val_loss_per_epoch,val_overlap_acc,val_residue_acc = val_one_epoch(model, criterion, val_loader, device, param['decoder_type'], result_type)  

            if 'class' in result_type:            
                # Storing performance locally
                train_acc.append(train_overlap_acc['average_overlap_acc'])
                train_loss.append(train_loss_per_epoch.item())
                val_acc.append(val_overlap_acc['average_overlap_acc'])
                val_loss.append(val_loss_per_epoch.item())

            #CONTINUE HERE ADD AVERAGE PER_RESIDUE_ACC TO DICT IN UTILS.PY AND THEN USE IT HERE ALSO ADD IT TO SAVE MODELS OPTION AND ADD OPTION TO LOG TO WANDB DIFFERENTLY
            if result_type == ['residue']:
                train_acc.append(train_per_residue_acc['average_per_residue_acc'])
                train_loss.append(train_loss_per_epoch.item())
                val_acc.append(val_per_residue_acc['average_per_residue_acc'])
                val_loss.append(val_loss_per_epoch.item())

  
            if param['tracking']:
                
                # Logging performance to weights and biases
                log_dict = {
                            'train/loss': train_loss_per_epoch,
                            'val/loss': val_loss_per_epoch
                           }
                
                # Add train and validation metrics to the log dictionary
                if 'class' in result_type:
                    add_metrics_to_log('train', train_overlap_acc, log_dict)
                    add_metrics_to_log('train', train_residue_acc, log_dict)
                    add_metrics_to_log('val', val_overlap_acc, log_dict)
                    add_metrics_to_log('val', val_residue_acc, log_dict)

                if 'residue' in result_type: 
                    add_metrics_to_log('train', train_per_residue_acc, log_dict)
                    add_metrics_to_log('val', val_per_residue_acc, log_dict)

                wandb.log(log_dict)
                """
                wandb.log({'train/loss': train_loss_per_epoch,
                          'train/top_acc': train_average_overlap_acc,
                          'train/sptm_overlap_acc': train_sptm_overlap_acc,
                          'train/tm_overlap_acc': train_tm_overlap_acc,
                          'train/sp_overlap_acc': train_sp_overlap_acc,
                          'train/glob_overlap_acc': train_glob_overlap_acc,
                          'train/beta_overlap_acc': train_beta_overlap_acc,
                          'train/sptm_residue_acc': train_sptm_residue_acc,
                          'train/tm_residue_acc': train_tm_residue_acc,
                          'train/sp_residue_acc': train_sp_residue_acc,
                          'train/glob_residue_acc': train_glob_residue_acc,
                          'train/beta_residue_acc': train_beta_residue_acc,
                          'val/loss': val_loss_per_epoch,
                          'val/top_acc': val_average_overlap_acc,
                          'val/sptm_overlap_acc': val_sptm_overlap_acc,
                          'val/tm_overlap_acc': val_tm_overlap_acc,
                          'val/sp_overlap_acc': val_sp_overlap_acc,
                          'val/glob_overlap_acc': val_glob_overlap_acc,
                          'val/beta_overlap_acc': val_beta_overlap_acc,
                          'val/sptm_residue_acc': val_sptm_residue_acc,
                          'val/tm_residue_acc': val_tm_residue_acc,
                          'val/sp_residue_acc': val_sp_residue_acc,
                          'val/glob_residue_acc': val_glob_residue_acc,
                          'val/beta_residue_acc': val_beta_residue_acc})
                 """
            # Saving best model per CV-fold
            if 'class' in result_type:
                if val_residue_acc['average_residue_acc'] > best_acc:
                    best_acc = val_residue_acc['average_residue_acc']
                    torch.save(model.state_dict(), f'{args.run}best_model.cv{k}.pt')

            elif result_type == ['residue']:
                if val_per_residue_acc['average_per_residue_acc'] > best_acc:
                    best_acc = val_per_residue_acc['average_per_residue_acc']
                    torch.save(model.state_dict(), f'{args.run}best_model.cv{k}.pt')
            """
            # USING THEIR METRICS
            train_loss_per_epoch, train_top_acc, train_sptm_top_acc, train_tm_top_acc, train_sp_top_acc, train_glob_top_acc, train_beta_top_acc = train_one_epoch(model, optimizer, criterion, train_loader, device)
            val_loss_per_epoch, val_top_acc, val_sptm_top_acc, val_tm_top_acc, val_sp_top_acc, val_glob_top_acc, val_beta_top_acc = val_one_epoch(model, criterion, val_loader, device)
            
            # Storing performance locally
            train_acc.append(train_top_acc)
            train_loss.append(train_loss_per_epoch.item())
            val_acc.append(val_top_acc)
            val_loss.append(val_loss_per_epoch.item())
            
            # Logging performance to weights and biases 
            wandb.log({'train/loss': train_loss_per_epoch,
                          'train/top_acc': train_top_acc,
                          'train/sptm_top_acc': train_sptm_top_acc,
                          'train/tm_top_acc': train_tm_top_acc,
                          'train/sp_top_acc': train_sp_top_acc,
                          'train/glob_top_acc': train_glob_top_acc,
                          'train/beta_top_acc': train_beta_top_acc,
                          'val/loss': val_loss_per_epoch,
                          'val/top_acc': val_top_acc,
                          'val/sptm_top_acc': val_sptm_top_acc,
                          'val/tm_top_acc': val_tm_top_acc,
                          'val/sp_top_acc': val_sp_top_acc,
                          'val/glob_top_acc': val_glob_top_acc,
                          'val/beta_top_acc': val_beta_top_acc})
            
            # Saving best model per CV-fold
            if val_top_acc > best_acc:
                best_acc = val_top_acc
                torch.save(model.state_dict(), f'{args.run}best_model.cv{k}.pt')
            """
            c += 1
            print(f'Finnished fold: {k}/{len(data_splits)}, Epoch: {c}/' + str(param['num_epoch']))
            
            # Saving last model
            if c == param['num_epoch']:
                torch.save(model.state_dict(), f'{args.run}last_model.cv{k}.pt')
        if param['tracking']:
            # End weights biases for every CV-fold to get 5 seperate learning curves 
            wandb.finish()

    print("Finished training.")
