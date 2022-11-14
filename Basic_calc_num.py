import numpy as np

class zerosf:
    
    #Função para o método da Bissecção
    def Mbiss(x1,x2,y,e):
        n=0
        #checa se os sinais de y(x1) e y(x2) são iguais
        #(se sim, provavelmente não há uma raiz em [x1,x2])
        if np.sign(y(x1))==np.sign(y(x2)):
            return print('Não há raiz nesse intervalo!')
        else:
            print('n :','x1 -','x2 -','xm')
            #enquanto a diferença de x1 e x2 for maior que a toleância
            #continuamos o método da bisseção
            while np.abs(x1-x2)>=e:
                #calcula x médio entre x1 e x2
                xm=(x1+x2)/2
                print(n,':',x1,'-',x2,'-',xm)
                #se y(x1) e y(x2) tiverem o mesmo sinal, a raiz deve se 
                #encontrar entre xm e x2
                if np.sign(y(x1))==np.sign(y(xm)):
                    #redefinimos x1
                    x1=xm
                else:
                    #senão redefinimos x2
                    x2=xm
                n=n+1
        return x1
    
    
    #Método de Newton-Raphson
    def MNR(x0,y,dy,e):
        n=0
        x=x0
        print('n :','x -','y(x) -','e')
        #enquanto y(x) não explodir
        while np.abs(y(x)*e)<1:
            print(n,':',x,'-',y(x),'-',y(x)/dy(x)/x)
            #quando o erro relativo for menor que a tolerância 
            #nós encontramos uma raiz
            if np.abs((y(x)/dy(x))/x)<e:
                return x
            #senão atualizamos x e continuamos
            else:
                x=x-y(x)/dy(x)
                n=n+1
        #se y(x) "explodir", é porque o método não achou uma raiz
        return print('Não converge!')
    
    
    #Método das secantes
    def Msec(x0,x1,y,e):
        n=0
        #admitindo os valores iniciais
        a=x0
        b=x1
        print('n :','x -','y(x)')
        #enquanto y(x) não explodir
        while np.abs(y(a)*e)<1:
            print(n,':',a,'-',y(a))
            #parar se y(a) está dentro da tolerância
            if np.abs(y(a))<e:
                return a
            else:
                #atualizar a
                a=a-(a-b)*y(a)/(y(a)-y(b))
                n=n+1
            print(n,':',b,'-',y(b))
            #parar se y(b) está dentro da tolerância
            if np.abs(y(b))<e:
                return b
            else:
                #atualizar b
                b=b-(b-a)*y(b)/(y(b)-y(a))
                n=n+1
        #se y(x) explodir, não achamos uma raiz
        return print('Não converge!')

class lineq:
    
    #critérios para a aplicação dos métodos

    #critério das linhas/diagonal dominante (falhando, ainda pode convergir)
    def linha(A):
        for i in range (len(A)):
            a=0
            for j in range (len(A[0])):
                if i!=j:
                    a=a+np.abs(A[i][j])
            if np.abs(A[i][i])<=a:
                return 0 #diverge
        return 1 #converge

    #critério de sassenfeld
    def sassy(A):
        b=np.zeros(len(A))
        for i in range (len(A)):
            for j in range (len(A)):
                if j<=i-1:
                    b[i]=b[i]+b[j]*np.abs(A[i][j])/np.abs(A[i][i])
                if j>=i+1:
                    b[i]=b[i]+np.abs(A[i][j])/np.abs(A[i][i])
            if b[i]>=1:
                return 0 #diverge
        return 1 #converge
    
    
    #Função que troca as linhas i e p da matriz A
    def troca(A,i,p):
        B=np.zeros((len(A),len(A[0])))
        for j in range (len(A)):
            B[j]=A[j]
            if j==i:
                B[i]=A[p]
            if j==p:
                B[p]=A[i]      
        return B

    #Método da Eliminação de Gauss
    def Egauss(A):
        print('Sendo a matriz aumentada:\n',A)
        #n é número de linhas em A
        n=len(A)
        B=A
        for i in range (0,n-1,1):
            a=0
            #Procuramos nessa coluna o maior coef. para usar como pivô
            for p in range (i,n,1):
                if np.abs(B[p][i])>a:
                    #ao achar um pivô, guardamos seu valor e a sua linha
                    a=B[p][i]
                    P=p
            #se não há nenhum pivô, não há solução.
            if a==0:
                return print('Não há pivô!')

            #trocamos a ordem das linhas.
            if P!=i:
                B=lineq.troca(B,P,i)
                print('\nTrocamos as linhas:',i+1,'e',P+1,'\n',B)
            
            #aqui fazemos a eliminação
            for j in range (i+1,n,1):
                if B[j][i]!=0:
                    m=B[j][i]/B[i][i]
                    for k in range (n+1):
                        B[j][k]=B[j][k]-m*B[i][k]
                    print('\nEliminamos o elemento',j+1,i+1,':\n',B)
    
        #checa se há solução
        if B[n-1][n-1]==0:
            return print('Não há solução única!')
    
        #construimos o vetor solução
        x=np.zeros(n)
        x[n-1]=B[n-1][n]/B[n-1][n-1]
        for i in range (n-1,-1,-1):
            x[i]=B[i][n]/B[i][i]
            for j in range (i+1,n,1):
                x[i]=x[i]-B[i][j]*x[j]/B[i][i]
        return print('\nA solução é x =',x)
    
    
    #Método Iterativo de Jacobi
    def Jacobi(A,b,x0,e,N):
        k=0
        n=len(b)
        print(k,'ª iteração:\nx=',x0)
        while k<N:
            #vetor solução
            x=np.zeros(n)
            y=np.zeros(n)
            #começamos a próxima iteração
            k=k+1
            for i in range (n):
                x[i]=b[i]/A[i][i]
                for j in range (n):
                    if j!=i:
                        x[i]=x[i]-A[i][j]*x0[j]/A[i][i]
                    #definimos o vetor 'erro'
                    y[j]=np.abs(x[j]-x0[j])
        
            #checa se alcançamos a precisão desejada
            if np.max(y)<e:
                return print('\n',k,'ª iteração:\nx=',x,'\nerro=',np.max(y))
        
            print('\n',k,'ª iteração:\nx=',x,'\nerro=',np.max(y))
            #atualizamos o valor do chute começamos de novo
            x0=x
        return
    
    #Método Gauss-Seidel
    def GSeidel(A,b,x0,e,N):
        k=0
        n=len(b)
        print(k,'ª iteração:\nx=',x0)
        while k<N:
            #vetor solução
            x=np.zeros(n)
            y=np.zeros(n)
            #começamos a próxima iteração
            k=k+1
            for i in range (n):
                x[i]=b[i]/A[i][i]
                for j in range (n):
                    #aqui usamos os valores de x calculados nessa iteração
                    if j<=i-1:
                        x[i]=x[i]-A[i][j]*x[j]/A[i][i]
                    #aqui usamos os valores de x calculados na última iteração
                    if j>=i+1:
                        x[i]=x[i]-A[i][j]*x0[j]/A[i][i]
                    #definimos o vetor 'erro'
                    y[j]=np.abs(x[j]-x0[j])
        
            #checa se alcançamos a precisão desejada
            if np.max(y)<e:
                return print('\n',k,'ª iteração:\nx=',x,'\nerro=',np.max(y))
        
            print('\n',k,'ª iteração:\nx=',x,'\nerro=',np.max(y))
            #atualizamos o valor do chute começamos de novo
            x0=x
        return

class numint:
    
    #método dos trapézios
    def Itrap(y,a,b,n):
        #com n intervalos, h é o tamanho de cada um
        h=(b-a)/n
        x=np.arange(a,b,h)
        I=0
        for i in range (len(x)-1):
            I=I+h*(y(x[i])+y(x[i+1]))/2  
        return I
    
    
    #Método de Simpson
    def Isimp(y,a,b,n):
        h=(b-a)/n
        x=np.arange(a,b,h)
        I=h*(y(a)+y(b))/3
        for i in range (1,n,2):
            I=I+4*h*y(x[i])/3
        for i in range (2,n,2):
            I=I+2*h*y(x[i])/3
        return I

class edo:
    
    #Euler
    def Eul1(f,y0,t0,tf,h):
        t_v=np.arange(t0,tf,h)
        y_v=np.zeros(len(t_v))
        t_v[0]=t0
        y_v[0]=y0
    
        for i in range (1,len(t_v),1):
            y_v[i]=y_v[i-1]+h*f(t_v[i-1],y_v[i-1])
            t_v[i]=t_v[i-1]+h
        return y_v   
    
    def Eul2(f,y0,z0,t0,tf,h):
        t_v=np.arange(t0,tf,h)
        y_v=np.zeros(len(t_v))
        z_v=np.zeros(len(t_v))
        t_v[0]=t0
        y_v[0]=y0
        z_v[0]=z0
    
        for i in range (1,len(t_v),1):
            y_v[i]=y_v[i-1]+h*z_v[i-1]
            z_v[i]=z_v[i-1]+h*f(t_v[i-1],y_v[i-1],z_v[i-1])
            t_v[i]=t_v[i-1]+h
        return y_v,z_v
    
    #Runge-Kutta
    def RKO1(f,y0,t0,tf,h):
        t_v=np.arange(t0,tf,h)
        y_v=np.zeros(len(t_v))
        t_v[0]=t0
        y_v[0]=y0
    
        for i in range (1,len(t_v),1):
            k1=h*f(t_v[i-1],y_v[i-1])
            k2=h*f(t_v[i-1]+h/2,y_v[i-1]+k1/2)
            k3=h*f(t_v[i-1]+h/2,y_v[i-1]+k2/2)
            k4=h*f(t_v[i-1]+h/2,y_v[i-1]+k3/2)
            y_v[i]=y_v[i-1]+(k1+2*k2+2*k3+k4)/6
            t_v[i]=t_v[i-1]+h
    
        return y_v
    
    
    def RKO2(f,y0,z0,t0,tf,h):
        t_v=np.arange(t0,tf,h)
        z_v=np.zeros(len(t_v))
        y_v=np.zeros(len(t_v))
        t_v[0]=t0
        z_v[0]=z0
        y_v[0]=y0
    
        for i in range (1,len(t_v),1):
            k1y=h*z_v[i-1]
            k1z=h*f(t_v[i-1],y_v[i-1],z_v[i-1])
            k2y=h*(z_v[i-1] + k1z/2)
            k2z=h*f(t_v[i-1]+h/2,y_v[i-1]+k1y/2,z_v[i-1]+k1z/2)
            k3y=h*(z_v[i-1] + k2z/2)
            k3z=h*f(t_v[i-1]+h/2,y_v[i-1]+k2y/2,z_v[i-1]+k2z/2)
            k4y=h*(z_v[i-1] + k3z/2)
            k4z=h*f(t_v[i-1]+h/2,y_v[i-1]+k3y/2,z_v[i-1]+k3z/2)
        
            y_v[i]=y_v[i-1]+(k1y+2*k2y+2*k3y+k4y)/6
            z_v[i]=z_v[i-1]+(k1z+2*k2z+2*k3z+k4z)/6
            t_v[i]=t_v[i-1]+h
        return y_v,z_v
