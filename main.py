import torch
import datetime

from config import args
from utils import seed_everything
from data.fgl_dataset import FGLDataset
from federated.trainer import NormalizeTrainer
from federated.fedgta.fedgta_server import FedGTAServer as FLServer
from federated.fedgta.fedgta_client import FedGTAClient as FLClient





if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    print(f"program start: {datetime.datetime.now()}")
    print(args,"\n")
    seed_everything(args.seed)
    device = torch.device('cuda:{}'.format(args.gpu_id) if (args.use_cuda and torch.cuda.is_available()) else 'cpu')
    dataset = FGLDataset(
        root=args.root,
        name=args.name,
        num_clients=args.num_clients,
        partition=args.partition,
        train=0.2,
        val=0.4,
        test=0.4,
        device=device
    )
    server = FLServer(
        dataset=dataset,
        args=args,
        device=device
    )
    clients = []
    for client_id in range(args.num_clients):
        client = FLClient(
            client_id=client_id,
            dataset=dataset,
            args=args,
            device=device
        )
        clients.append(client)
    trainer = NormalizeTrainer(
        server=server,
        clients=clients,
        args=args,
        device=device
    )
    trainer.normalize_train()

