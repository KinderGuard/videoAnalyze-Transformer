import torch
import torch.nn.functional as F #nn과 같은 모듈이 모아져 있지만 함수의 input으로 반드시 연산이 되어야 하는 값을 받음
from torch import nn #Deep learning model에 필요한 모듈이 모아져 있는 패키지
import numpy as np
import math
import torchvision
from torch.autograd import Variable


# Standard 2 layerd FFN of transformer
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.3):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

        nn.init.normal(self.linear_1.weight, std=0.001)
        nn.init.normal(self.linear_2.weight, std=0.001)

    # 모델이 학습데이터 입력받아서 forward propagation 진행시키는 함수
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# standard NORM layer of Transformer
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6, trainable=True):
        super(Norm, self).__init__()
        self.size = int(d_model)
        # create two learnable parameters to calibrate normalisation
        if trainable:
            self.alpha = nn.Parameter(torch.ones(self.size))
            self.bias = nn.Parameter(torch.zeros(self.size))
        else:
            self.alpha = nn.Parameter(torch.ones(self.size), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(self.size), requires_grad=False)
        self.eps = eps

    # 모델이 학습데이터 입력받아서 forward propagation 진행시키는 함수
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


# Standard positional encoding (addition/ concat both are valid)
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        #nn.Module.__init__()실향하는 코드
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        #지정된 위치에 삽입되어 있는 크기 1의 차원을 가진 새 tensor반환
        pe = pe.unsqueeze(0)
        #weight와 bias 생성
        self.register_buffer('pe', pe)

    # 모델이 학습데이터 입력받아서 forward propagation 진행시키는 함수
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        batch_size = x.size(0)
        num_feature = x.size(2)
        spatial_h = x.size(3)
        spatial_w = x.size(4)

        #Variable로 forward pass계산, gradient계산 필요 없음
        #Variable은 backpropagation을 수행하는 backward method 제공함
        #variables는 tensor를 감싸고, gradients가 역으로 흐를 수 있도록 tensor사이에 연산 체인을 구축함
        z = Variable(self.pe[:, :seq_len], requires_grad=False)
        z = z.unsqueeze(-1).unsqueeze(-1)
        #dimension이 더 큰 크기로 확장된 해당 텐서의 새 view를 반환
        z = z.expand(batch_size, seq_len, num_feature, spatial_h, spatial_w)
        x = x + z
        return x


# standard attention layer
def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.sum(q * k, -1) / math.sqrt(d_k)
    # scores : b, t
    scores = F.softmax(scores, dim=-1)
    scores = scores.unsqueeze(-1).expand(scores.size(0), scores.size(1), v.size(-1))
    # scores : b, t, dim
    output = scores * v
    output = torch.sum(output, 1)
    if dropout:
        output = dropout(output)
    return output

#Action Transformer Unit
class TX(nn.Module):
    def __init__(self, d_model=64, dropout=0.3):
        super(TX, self).__init__()
        self.d_model = d_model
        # no of head has been modified to encompass : 1024 dimension
        #nn.Dropout(p: float = 0.5, inplace: bool = False): training 중 베르누이 분포의 표본을 사용하여 확률 p로 입력 tensor의 일부 요소를 부작위로 0으로 설정함
        #성공률이 p인 실험에서 성공이면 X = 1, 실패이면 X = 0이라 할 때, 확률변수 X의 확률분포를 모수 p인 베르누이 분포라 함.
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        #FeedForward모듈로~
        self.ff = FeedForward(d_model, d_ff=int(d_model / 2))

    # 모델이 학습데이터 입력받아서 forward propagation 진행시키는 함수
    def forward(self, q, k, v, mask=None):
        # q: (b , dim )
        b = q.size(0)
        t = k.size(1)
        dim = q.size(1)
        q_temp = q.unsqueeze(1)
        q_temp = q_temp.expand(b, t, dim)
        # q,k,v : (b, t , d_model=1024 // 16 )
        #attention 모듈로~
        A = attention(q_temp, k, v, self.d_model, mask, self.dropout)
        # A : (b , d_model=1024 // 16 )
        q_ = self.norm_1(A + q)
        #FeedForward한 모델을 dropout하고 norm한 게 새로운 query다~
        new_query = self.norm_2(q_ + self.dropout_2(self.ff(q_)))
        return new_query

#Tail에서 호출함
class Block_head(nn.Module):
    def __init__(self, d_model=64, dropout=0.3):
        super(Block_head, self).__init__()
        #TX모듈로~
        self.T1 = TX()
        self.T2 = TX()
        self.T3 = TX()

    # 모델이 학습데이터 입력받아서 forward propagation 진행시키는 함수
    def forward(self, q, k, v, mask=None):
        q = self.T1(q, k, v)
        q = self.T2(q, k, v)
        q = self.T3(q, k, v)
        return q

#q,k,v 등장
class Tail(nn.Module):
    def __init__(self, num_classes, num_frames, head=16):
        super(Tail, self).__init__()
        self.spatial_h = 7
        self.spatial_w = 4
        self.head = head
        self.num_features = 2048
        self.num_frames = num_frames
        #linear의 input sample size
        self.d_model = self.num_features / 2
        self.d_model = int(self.d_model)
        self.d_k = self.d_model // self.head
        #4D이상의 input에 Batch Normalization적용
        #Batch Normalization: 내부 공변량(covariate)를 줄여 심층 네트워크 교육을 가속화할 수 있음
        #공변량: 실험결과의 독립변수 이외에 종속변수에 영향을 줄 수 있는 잡음인자(실험자가 통제하고자 하는 변수)
        self.bn1 = nn.BatchNorm2d(self.num_features)
        #Norm Layer
        self.bn2 = Norm(self.d_model, trainable=False)

        #PositionalEncoder 모듈로~
        self.pos_embd = PositionalEncoder(self.num_features, self.num_frames)
        self.Qpr = nn.Conv2d(self.num_features, self.d_model, kernel_size=(7, 4), stride=1, padding=0, bias=False)

        self.head_layers = []
        for i in range(self.head):
            #Block_head모듈로~
            self.head_layers.append(Block_head())

        self.list_layers = nn.ModuleList(self.head_layers)
        #input데이터에 선형 변환 적용: y = xA^T + b
        self.classifier = nn.Linear(self.d_model, num_classes)
        # resnet style initialization
        nn.init.kaiming_normal(self.Qpr.weight, mode='fan_out')
        nn.init.normal(self.classifier.weight, std=0.001)
        # nn.init.constant(self.classifier.bias, 0)

        nn.init.constant(self.bn1.weight, 1)
        nn.init.constant(self.bn1.bias, 0)

    #모델이 학습데이터 입력받아서 forward propagation 진행시키는 함수
    def forward(self, x, b, t):
        #BatchNorm2d실행
        x = self.bn1(x)
        # stabilizes the learning
        x = x.view(b, t, self.num_features, self.spatial_h, self.spatial_w)
        #PositionalEncoder 모듈로~
        x = self.pos_embd(x)
        x = x.view(-1, self.num_features, self.spatial_h, self.spatial_w)
        x = F.relu(self.Qpr(x))
        # x: (b,t,1024,1,1) since its a convolution: spatial positional encoding is not added
        # paper has a different base (resnet in this case): which 2048 x 7 x 4 vs 16 x 7 x 7
        x = x.view(-1, t, self.d_model)
        #Norm Layer실행
        x = self.bn2(x)
        # stabilization
        q = x[:, t / 2, :]  # middle frame is the query
        v = x  # value
        k = x  # key

        q = q.view(b, self.head, self.d_k)
        k = k.view(b, t, self.head, self.d_k)
        v = v.view(b, t, self.head, self.d_k)

        #dim1, dim2 교환
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        #  q: b, 16, 64
        #  k,v: b, 16, 10 ,64
        outputs = []
        for i in range(self.head):
            outputs.append(self.list_layers[i](q[:, i], k[:, i], v[:, i]))

        #주어진 일련의 seq tensor들을 주어진 차원으로 연결
        f = torch.cat(outputs, 1)

        #특정 dim에 Lp normalization수행
        f = F.normalize(f, p=2, dim=1)
        # F.norma
        if not self.training:
            return f
        #f에 대한 선형 변환 수행
        y = self.classifier(f)
        return y, f


# base is resnet
# Tail is the main transformer network
class Semi_Transformer(nn.Module):
    def __init__(self, num_classes, seq_len):
        super(Semi_Transformer, self).__init__()
        #residual learning framework(이미지 분류용 CNN모델)ReLU사용하는 거~!
        #pretrained model = True -> weight를 캐시 디렉토리에 다운로드함
        resnet50 = torchvision.models.resnet50(pretrained=True)
        #torch.nn.Sequential(*args: Any)순차적 컨테이너. 모듈들이 들어온 순서대로 constructor에 추가됨
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.tail = Tail(num_classes, seq_len)

    # 모델이 학습데이터 입력받아서 forward propagation 진행시키는 함수
    def forward(self, x):
        b = x.size(0)#batch size
        t = x.size(1)#..frame수??
        #.view(): 텐서의 형상 변환(reshape)이나 차원을 늘리거나 줄일 때 사용
        #dimension, size
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        #resNet어쩌구 뭔가 해놓기
        x = self.base(x)
        # x: (b,t,2048,7,4)
        #모듈 Tail로~
        return self.tail(x, b, t)
