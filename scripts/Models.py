from typing import Optional, Set, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphein.protein.tensor.data import ProteinBatch
from loguru import logger as log
from torch_geometric.data import Batch
from torch_scatter import scatter_add
from proteinworkshop.models.graph_encoders.layers import gear_net
from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.types import EncoderOutput
from torch_geometric.utils import unbatch
from torch.nn.utils.rnn import pack_sequence, unpack_sequence

class GearNet(nn.Module):
    def __init__(
        self,
        input_dim = 23,
        num_relation = 2,
        num_layers = 6,
        emb_dim = 16,
        short_cut = True,
        concat_hidden = True,
        batch_norm = True,
        num_angle_bin = 8,
        activation = "relu",
        pool = "sum",
    ):
        """Initializes an instance of the GearNet model.

        :param input_dim: Dimension of the input node features
        :type input_dim: int
        :param num_relation: Number of edge types
        :type num_relation: int
        :param num_layers: Number of layers in the model
        :type num_layers: int
        :param emb_dim: Dimension of the node embeddings
        :type emb_dim: int
        :param short_cut: Whether to use short cut connections
        :type short_cut: bool
        :param concat_hidden: Whether to concatenate hidden representations
        :type concat_hidden: bool
        :param batch_norm: Whether to use batch norm
        :type batch_norm: bool
        :param num_angle_bin: Number of angle bins for edge message passing.
            If ``None``, edge message passing is not disabled.
        :type num_angle_bin: Optional[int]
        :param activation: Activation function to use, defaults to "relu"
        :type activation: str, optional
        :param pool: Pooling operation to use, defaults to "sum"
        :type pool: str, optional
        """
        super().__init__()
        # Base parameters
        self.num_relation = num_relation
        self.input_dim = input_dim
        self.num_layers = num_layers
        # Edge message passing layers
        # If not None, this enables Edge Message passing
        self.num_angle_bin = num_angle_bin
        self.edge_input_dim = self._get_num_edge_features()
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        n_hid = [emb_dim] * num_layers

        self.dims = [self.input_dim] + n_hid
        self.activations = [getattr(F, activation) for _ in n_hid]
        self.batch_norm = batch_norm

        # Initialise Node layers
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                gear_net.GeometricRelationalGraphConv(
                    input_dim=self.dims[i],
                    output_dim=self.dims[i + 1],
                    num_relation=self.num_relation,
                    edge_input_dim=self.edge_input_dim,  # None,
                    batch_norm=batch_norm,
                    activation=self.activations[i],
                )
            )

        if self.num_angle_bin:
            log.info("Using Edge Message Passing")
            self.edge_input_dim = self._get_num_edge_features()
            self.edge_dims = [self.edge_input_dim] + self.dims[:-1]
            self.spatial_line_graph = gear_net.SpatialLineGraph(
                self.num_angle_bin
            )
            self.edge_layers = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(
                    gear_net.GeometricRelationalGraphConv(
                        self.edge_dims[i],
                        self.edge_dims[i + 1],
                        self.num_angle_bin,
                        None,
                        batch_norm=self.batch_norm,
                        activation=self.activations[i],
                    )
                )
        # Batch Norm
        if self.batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))

        # Readout
        self.readout = get_aggregation(pool)

    @property
    def required_batch_attributes(self) -> Set[str]:
        """Required batch attributes for this encoder.

        - ``x`` Positions (shape ``[num_nodes, 3]``)
        - ``edge_index`` Edge indices (shape ``[2, num_edges]``)
        - ``edge_type`` Edge types (shape ``[num_edges]``)
        - ``edge_attr`` Edge attributes (shape ``[num_edges, num_edge_features]``)
        - ``num_nodes`` Number of nodes (int)
        - ``batch`` Batch indices (shape ``[num_nodes]``)

        :return: Set of required batch attributes.
        :rtype: Set[str]
        """
        return {
            "x",
            "edge_index",
            "edge_type",
            "edge_attr",
            "num_nodes",
            "batch",
        }

    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Implements the forward pass of the GearNet encoder.

        Returns the node embedding and graph embedding in a dictionary.

        :param batch: Batch of data to encode.
        :type batch: Union[Batch, ProteinBatch]
        :return: Dictionary of node and graph embeddings. Contains
            ``node_embedding`` and ``graph_embedding`` fields. The node
            embedding is of shape :math:`(|V|, d)` and the graph embedding is
            of shape :math:`(n, d)`, where :math:`|V|` is the number of nodes
            and :math:`n` is the number of graphs in the batch and :math:`d` is
            the dimension of the embeddings.
        :rtype: EncoderOutput
        """
        hiddens = []
        batch.edge_weight = torch.ones(
            batch.edge_index.shape[1], dtype=torch.float, device=batch.x.device
        )
        layer_input = batch.x
       
        batch.edge_index = torch.cat([batch.edge_index, batch.edge_type])
        batch.edge_feature = self.gear_net_edge_features(batch)
        
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(batch)
            line_graph.edge_weight = torch.ones(
                line_graph.edge_index.shape[1],
                dtype=torch.float,
                device=batch.x.device,
            )
            edge_input = line_graph.x.float()

        for i in range(len(self.layers)):
            hidden = self.layers[i](batch, layer_input)    
            
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)
                edge_weight = batch.edge_weight.unsqueeze(-1)
                # node_out = graph.edge_index[:, 1] * self.num_relation + graph.edge_index[:, 2]
                node_out = (
                    batch.edge_index[1, :] * self.num_relation
                    + batch.edge_index[2, :]
                )
                update = scatter_add(
                    edge_hidden * edge_weight,
                    node_out,
                    dim=0,
                    dim_size=batch.num_nodes * self.num_relation,
                )
                update = update.view(
                    batch.num_nodes, self.num_relation * edge_hidden.shape[1]
                )
                update = self.layers[i].linear(update)
                update = self.layers[i].activation(update)
                hidden = hidden + update

                edge_input = edge_hidden
            if self.batch_norm:
                hidden = self.batch_norms[i](hidden)
            hiddens.append(hidden)
            layer_input = hidden
        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        return EncoderOutput(
            {
                "node_embedding": node_feature,
                "graph_embedding": self.readout(node_feature, batch.batch),
            }
        )


    def _get_num_edge_features(self) -> int:
        """Compute the number of edge features."""
        seq_dist = 1
        dist = 1
        return self.input_dim * 2 + self.num_relation + seq_dist + dist

    def gear_net_edge_features(
        self, b: Union[Batch, ProteinBatch]
    ) -> torch.Tensor:
        """Compute edge features for the gear net encoder.

        - Concatenate node features of the two nodes in each edge
        - Concatenate the edge type
        - Compute the distance between the two nodes in each edge
        - Compute the sequence distance between the two nodes in each edge

        :param b: Batch of data to encode.
        :type b: Union[Batch, ProteinBatch]
        :return: Edge features
        :rtype: torch.Tensor
        """
        u = b.x[b.edge_index[0]]
        v = b.x[b.edge_index[1]]
        edge_type = F.one_hot(b.edge_type.view(1,-1), self.num_relation)[0]
        dists = torch.pairwise_distance(
            b.pos[b.edge_index[0]], b.pos[b.edge_index[1]]
        ).unsqueeze(1)
        seq_dist = torch.abs(b.edge_index[0] - b.edge_index[1]).unsqueeze(1)
        return torch.cat([u, v, edge_type, seq_dist, dists], dim=1)
    
class DenseDecoder(torch.nn.Module):
   def __init__(self,n_classes=6,hidden_dim = 16):
      super(DenseDecoder, self).__init__()
      #self.linear1 = linear(in_features=20,out_channels=n_classes)
      self.ll = torch.nn.Linear(hidden_dim,n_classes,bias=True)
   def forward(self,x, batch): #MAYBE REMOVE BATCH OR FIX FOR LATER
      #print("Decoder layer received a shape: ",x["node_embedding"].shape)
      return self.ll(x["node_embedding"])

class LSTMDecoder(torch.nn.Module):
   def __init__(self,n_classes=6, GCN_hidden_dim = 32, LSTM_hidden_dim = 32, dropout = 0.0, type="LSTMB",LSTMnormalization = False,lstm_layers=1):
      super(LSTMDecoder, self).__init__()
      #self.linear1 = linear(in_features=20,out_channels=n_classes)
      if(type=="LSTMO"):
         bid = False
         self.proj = torch.nn.Linear(LSTM_hidden_dim,n_classes) #project to per class sequence
      elif(type=="LSTMB"):
         bid = True
         self.proj = torch.nn.Linear(LSTM_hidden_dim*2,n_classes) #project to per class sequence

      if(LSTMnormalization):
         self.norm = torch.nn.LayerNorm(GCN_hidden_dim)
      else:
         self.norm = None
      self.LSTM = torch.nn.LSTM(input_size = GCN_hidden_dim,hidden_size=LSTM_hidden_dim,dropout=dropout,bidirectional=bid,num_layers=lstm_layers)


      self.init_LSTM()

   def init_LSTM(self):
      # for weight in self.LSTM._all_weights:
      #    if("weight" in weight):
      #       torch.nn.init.xavier_normal_(getattr(self.LSTM,weight))
      #    if("bias" in weight):
      #       torch.nn.init.normal_(getattr(self.LSTM,weight))
      for name, param in self.LSTM.named_parameters():
         if 'bias' in name:
            torch.nn.init.constant_(param, 0.0)
         elif 'weight' in name:
            torch.nn.init.xavier_normal_(param)
      print("Initialized LSTM layers")

   def forward(self,x,batch):
      if(self.norm!=None):
         x_in = self.norm(x["node_embedding"]) #equivalent to batch norm
      else:
         x_in = x["node_embedding"]

      #also return list of labels 
      

      #create a packed sequence object 
      unbatched = list(unbatch(x_in,batch)) #step 1: unbatch
      #print("Number of unbatched elements: ",len(unbatched))
      #print("Each with shape: ")
      #for i, x in enumerate(unbatched): print(f"\t protein {i}: {x.shape}")
      pseq = pack_sequence(unbatched,enforce_sorted=False) #step two: pack (includes zero-padding)

      #print("LSTM INPUT SHAPE: ",x_in.shape)
      #lstmoutput, hidden = self.LSTM(x_in) #without packed sequence
      lstmoutput, _ = self.LSTM(pseq)
      X = unpack_sequence(lstmoutput) #list of tensors
      output = []
      #protein_lengths = []
      for x in X: #can't use batch-dimension, as proteins are of different lengths!
         out = self.proj(x)
         output.append(out)
         #protein_lengths.append(out.shape[0])

      
      return output#, protein_lengths #list of tensors

class GraphEncDec(torch.nn.Module):
   def __init__(self, featuriser, n_classes=6,hidden_dim_GCN = 32, decoder_type='linear', LSTM_hidden_dim = 0, dropout = 0.0, LSTMnormalization = False,lstm_layers=1, type = 'LSTMB', encoder_concat_hidden = False, num_relation=1, input_dim=23):
      super(GraphEncDec,self).__init__()
      self.featuriser = featuriser
      self.encoder = GearNet(emb_dim = hidden_dim_GCN, concat_hidden = encoder_concat_hidden, num_relation=num_relation, input_dim=input_dim)
      #self.init_lazy_layers()  #initialize weights of LazyLinear layers by forwarding a random batch
      if decoder_type == 'linear':
          if self.encoder.concat_hidden == False:
              self.decoder =  DenseDecoder(hidden_dim=hidden_dim_GCN,n_classes=n_classes)
          elif self.encoder.concat_hidden == True:
              self.decoder =  DenseDecoder(hidden_dim=hidden_dim_GCN * self.encoder.num_layers,n_classes=n_classes)
      elif decoder_type == 'lstm':
          if self.encoder.concat_hidden == False:
              self.decoder = LSTMDecoder(n_classes=n_classes,GCN_hidden_dim=hidden_dim_GCN, LSTM_hidden_dim = LSTM_hidden_dim, dropout = dropout, type=type,LSTMnormalization=LSTMnormalization,lstm_layers=lstm_layers)   
          elif self.encoder.concat_hidden == True:
              self.decoder = LSTMDecoder(n_classes=n_classes,GCN_hidden_dim=hidden_dim_GCN * self.encoder.num_layers, LSTM_hidden_dim = LSTM_hidden_dim, dropout = dropout, type=type,LSTMnormalization=LSTMnormalization,lstm_layers=lstm_layers)
   def forward(self,X):

#      print(f'X input dimension to model: {X.size()}')
      X = self.featuriser(X)

#      print(f'X dimension after featuriser : {X.size()}')
      protein_lengths = torch.bincount(X.batch).tolist()

      embeddings = self.encoder(X) #returns a dict containing node and graph embeddings
      #print("Encoder outputs: ",embeddings.keys())
      
#      print("Input shape to decoder:", embeddings["node_embedding"].size())
      predictions = self.decoder(embeddings, X.batch) #REMOVE BATCH FOR DENSE, FIX FOR AUTOMATION LATER 
  
#      print(f'Decoder: {self.decoder}')

#      print(f'Output shape of decoder: {predictions[0].size()}')
      return predictions, protein_lengths
