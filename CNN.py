import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# parameters
learning_rate = 0.001
training_epochs = 10000000
batch_size = 100

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

class CNN(torch.nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        #첫번째 layer 생성
        self.layer1 = torch.nn.Sequential( torch.nn.Conv2d(1, 32, kernel_size = 3, stride =1, padding = 1),
            #ReLU함수에 통과시킨다.
            torch.nn.ReLU(),torch.nn.MaxPool2d(kernel_size = 2, stride = 2))
            
        #두 번째 layer 생성, 이번에는 output filter이 64개이다.
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride =1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2))
        #fully connected layer 생성
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias = True) #보충설명1
        #가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)
   
   #모델 설계 후 데이터셋이 layer들을 통과할 수 있게 함.
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) #보충설명2
        out = self.fc(out)
        return out
    
# instantiate CNN model
model = CNN().to(device)
# define cost/loss & optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)    # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        '''
        우리가 만든 CNN을 model로 instantiate 해주었는데,
        model에 우리의 training set을 대입해 본다.
        '''
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

print('Learning Finished!')
