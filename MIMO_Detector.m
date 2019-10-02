%%%%%%%%%%%%%%%% y = Hx+n ����MIMO������ %%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%% ����������� %%%%%%%%%%%%%%%%%%%%%
clear;
close;
Nr = 8;    % ��������
Nt = 4;    % ��������
M = 4;    % ���ƽ�����ʹ����λ����
K = 1e5;  %ÿ���û�����K������
SNR = 0:5:35;
bit_number = Nt*K*log2(M);
OAMP_iter = 4;   % OAMP�㷨��������
is_ML = 1; % �Ƿ�ʹ��ML�����1��ʹ�� 0�������� ��һ�㲻ʹ�ã�ML����ᵼ�·���ʱ�������

%%
%%%%%%%%%%%%%%%%%%%%%% �����Ȼ���״̬�� %%%%%%%%%%%%%%%%%%%%%%%%
PSK_table = pskmod(0:M-1,M);
if is_ML ==1  %���ʹ��ML��⣬is_ML == 1
    for i = 0:power(M,Nt)-1
        tempx = 0;
        tempy = 0;
        for j = 1:Nt
            temp = tempx*tempy';  
            z(j,i+1) = floor((i-temp)/power(M,Nt-j));                            
            tempx(j+1) = z(j,i+1);
            tempy(j+1) = power(M,Nt-j);      
        end   
    end
    combin_ML = PSK_table(z+1); %���ܵķ�����ϣ�ÿһ�б�ʾһ�����
end

%%
%%%%%%%%%%%%%%%%%%%%%%% ʱ�����ŵ����ŵ�����ͷ����������� %%%%%%%%%%%%%%
H = randn(Nr, Nt,K)+1i*randn(Nr, Nt,K);         %��ƽ����mean(H.*conj(H),'all')

%��������˺ͽ��ն˵Ŀռ���ؾ���Rt��Rr(Steepest Descent Method Based Soft-Output Detection for Massive MIMO Uplink Pages 274)
cesai_r = 0.6;
cesai_t = 0.2;
theta = pi/2;
for i = 1:Nr
    for k = 1:Nr
        if i <= k
            R_r(i,k)=(cesai_r*exp(1i*theta))^(k-i);
        else
            R_r(i,k)=conj((cesai_r*exp(1i*theta))^(i-k));
        end
    end
end
for i = 1:Nt
    for k = 1:Nt
        if i <= k
            R_t(i,k)=(cesai_r*exp(1i*theta))^(k-i);
        else
            R_t(i,k)=conj((cesai_r*exp(1i*theta))^(i-k));
        end
    end
end

for i = 1:K 
    H(:,:,i) = sqrtm(R_r)*H(:,:,i)*sqrtm(R_t);
end

dataIn = randi([0,M-1],Nt,K);
dataMod = pskmod(dataIn,M);
x = dataMod;   %��ƽ����mean(x.*conj(x),'all')
for i = 1:K
    y(:,i)=H(:,:,i)*x(:,i);
end
power_rx = trace(y*y')/(Nr*K);
%%
ser_ZF = zeros(1,length(SNR));
ser_MMSE = zeros(1,length(SNR));
ser_ML = zeros(1,length(SNR));
ser_OAMP = zeros(1,length(SNR));
%%%%%%%%%%%%%%%%%%%% ���治ͬ������²�ͬ��ⷽ�������� %%%%%%%%%%%%%%%%
for i = 1:length(SNR)
    i
    snr = SNR(i)-10*log10(power_rx);
    sigma2 = 10^(-snr/10);
    noise = (randn(Nr,K)+1i*randn(Nr,K))*sqrt(sigma2/2);
    y_noise = y+noise;
    
    %%
    %%%%%%%%%%%%%%% ZF��� %%%%%%%%%%%%%%%%%%%%%%%%%
    tic
    for j = 1:K
        x_ZF(:,j) = inv(H(:,:,j)'*H(:,:,j))*H(:,:,j)'*y_noise(:,j);
    end
    dataOut_ZF = pskdemod(x_ZF,M);
    num_error_ZF = sum(dataOut_ZF~=dataIn,'all');
    ser_ZF(i) = num_error_ZF/(Nt*K);
    time_ZF = toc
    
    %%
    %%%%%%%%%%%%%%%%%%%% MMSE��� %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic 
    for j=1:K
        x_MMSE(:,j) = inv((H(:,:,j)'*H(:,:,j))+sigma2*diag(ones(1,Nt)))*(H(:,:,j)'*y_noise(:,j));
    end
    dataOut_MMSE = pskdemod(x_MMSE,M);
    num_error_MMSE = sum(dataOut_MMSE~=dataIn,'all');
    ser_MMSE(i) = num_error_MMSE/(Nt*K);
    time_MMSE = toc
    %%
    %%%%%%%%%%%%%%%%%%%% ML��� %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic
    if is_ML==1
        state_table = combin_ML;
        for j = 1:K
            distance_square = zeros(1,power(M,Nt));
            for k = 1:power(M,Nt)
                distance_square(1,k) = sum((y_noise(:,j)-H(:,:,j)*state_table(:,k)).*conj((y_noise(:,j)-H(:,:,j)*state_table(:,k))),1);
            end
            [~,I] = min(distance_square);
            x_ML(:,j) = state_table(:, I);
        end
        dataOut_ML = pskdemod(x_ML,M);
        num_error_ML = sum(dataOut_ML~=dataIn,'all');
        ser_ML(i) = num_error_ML/(Nt*K);
    end     
    time_ML = toc
    %%
    %%%%%%%%%%%%%%%%%%% OAMP��� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%% ������Model-Driven Deep learning .....%%%%%%%
    tic
    for j=1:K
        tau_t_square = 1;
        xt=zeros(Nt,1);
        for k = 1:OAMP_iter
            vt_square = abs(((y(:,j)-H(:,:,j)*xt)'*(y(:,j)-H(:,:,j)*xt)-Nr*sigma2)/trace(H(:,:,j)'*H(:,:,j)));
            Wt_est   = vt_square*H(:,:,j)'*inv(vt_square*H(:,:,j)*H(:,:,j)'+sigma2*diag(ones(1,Nr)));
            Wt = Nt/trace(Wt_est*H(:,:,j))*Wt_est;
            rt = xt+Wt*(y(:,j)-H(:,:,j)*xt);
            if k >=2
                Bt = diag(ones(1,Nt))-Wt*H(:,:,j);
                tau_t_square = 1/Nt*trace(Bt*Bt')*vt_square+1/Nt*trace(Wt*sigma2*diag(ones(1,Nr))*Wt');
            end
            for n = 1:Nt
                temp1 = 0;
                temp2 = 0;
                for m = 1:M
                    temp1 = temp1 + PSK_table(m)*1/(2*pi*tau_t_square)*exp(-abs((PSK_table(m)-rt(n)))^2/(2*tau_t_square))*1/M;
                    temp2 = temp2 + 1/(2*pi*tau_t_square)*exp(-abs((PSK_table(m)-rt(n)))^2/(2*tau_t_square))*1/M;
                end
                xt(n,1)=temp1/temp2;
            end    
        end
        x_OAMP(:,j) = xt;   
    end
    dataOut_OAMP = pskdemod(x_OAMP,M);
    num_error_OAMP = sum(dataOut_OAMP~=dataIn,'all');
    ser_OAMP(i) = num_error_OAMP/(Nt*K);
    time_OAMP = toc
end

%%
%%%%%%%%%%%%%%%%%%%%%%%% ��ͼ %%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
semilogy(SNR,ser_ZF,'r-','DisplayName','ZF���');
hold on;
semilogy(SNR,ser_MMSE,'g--','DisplayName','MMSE���');
semilogy(SNR,ser_ML,'b-.','DisplayName','ML���');
semilogy(SNR,ser_OAMP,'k-.','DisplayName','OAMP���');
grid on
xlabel('SNR[dB]');ylabel('SER');title('MIMO Detection Methods');
legend;

    
    