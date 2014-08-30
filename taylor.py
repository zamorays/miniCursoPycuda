# -*- coding: utf-8 -*- 

import numpy as np
import matplotlib.pyplot as plt

class taylorDiff(object):
    
    def __init__(self, jet=0.0, order=15):
        self.jet = np.array(jet)
        self.order = int(order)
        if self.jet.size != order+1:
	  self.jet.resize(int(order)+1) # len mayor a order en 1
    
    #Representaciones
    
    def __repr__(self): #Convention
        return "taylorDiff({},{})".format(repr(self.jet),repr(self.order))
    
    def __str__(self): #Pretty form
        s = ''
        for i in range(0, len(self.jet)):
            if self.jet[i] != 0:
                s += ' + %g*x^%d' % (self.jet[i], i)
        # Fix layout
        s = s.replace('+ -', '- ')
        s = s.replace('x^0', '1')
        s = s.replace(' 1*', ' ')
        s = s.replace('x^1 ', 'x ')
        #s = s.replace('x^1', 'x') # will replace x^100 by x^00
        if s[0:3] == ' + ':  # remove initial +
            s = s[3:]
        if s[0:3] == ' - ':  # fix spaces for initial -
            s = '-' + s[3:]
        return s
        
    def _repr_html_(self):
        pass
    
    def _repr_latex_(self):
        s = '$'
        for i in range(0, len(self.jet)):
            if self.jet[i] != 0:
                s += ' + %g\cdot x^{%d}' % (self.jet[i], i)
        # Fix layout
        s = s.replace('+ -', '- ')
        s = s.replace('x^0', '1')
        s = s.replace(' 1 ', ' ')
        s = s.replace('x^1 ', 'x ')
        #s = s.replace('x^1', 'x') # will replace x^100 by x^00
        if s[0:3] == '+':  # remove initial +
            s = s[3:]
        if s[0:3] == ' - ':  # fix spaces for initial -
            s = '-' + s[3:]
        return s+'$'
        
    #Horners RULE
    def __call__(self,x0): # OJO con sobrecarga de operadores jala para vectores, matrices, etc !!!!!!
      s = self.jet[-1]
      for i in range(1,self.order+1):
	s = self.jet[-(i+1)] + s*x0
	
      return s
      
    def draw(self, npoint=100, xdomain = [-1.,1.],mk='o',ls = '-*-',extraF=None):
      rnd = np.random.uniform(0.,1.,3) #Color
      lbl = self.__str__()
      if len(lbl) >= 60:
	lbl = lbl[0:60]+'...'
      plotF = plt.figure(1,figsize=(12,8),dpi=300)
      xgridp = np.linspace(xdomain[0],xdomain[1],npoint)
      fxgridp = self.__call__(xgridp)
      plt.plot(xgridp,fxgridp,color=(rnd[0],rnd[1],0.5),label=lbl,marker=mk,linestyle=ls)
      plt.xlabel('$x$', fontsize=20)
      plt.ylabel('$P_{taylor} ( x )$',fontsize=20)
      if extraF != None:
	plt.plot(xgridp,extraF(xgridp),color=(rnd[0],rnd[1],rnd[2]),label='Exact P(x)')
      
      plt.legend(loc=2)
    
    def set_order(self,orden=25):
      self.__init__(self.jet,orden)
      
    def cut_off(self,tol=1.e-15):
      nc = self.jet.nonzero()[0]
      idcero = None
      for i in range(nc.size-1):
	reldiff = abs(self.jet[nc[i+1]])/abs(self.jet[nc[i]])
	if reldiff <= tol:
	  self.jet[nc[i+1]] = 0.
	  idcero = nc[i+1]
	  break
	  
      if idcero != None:
	self.jet[idcero:] = 0.
      
	
    # Operaciones básicas
    def __add__(self, otro):
	try:
	  if self.order == otro.order:
	    return taylorDiff(self.jet + otro.jet,otro.order)
	  else:
	    orden = max(self.order,otro.order)
	    self.jet.resize(orden+1)
	    otro.jet.resize(orden+1)
	    self.order=orden
	    otro.order=orden
	    return taylorDiff(self.jet + otro.jet ,orden)
	except:
	  return self + taylorDiff(otro)
	  
    def __radd__(self, otro):
        return self + otro
        
    def __neg__(self):
	return taylorDiff(-self.jet,self.order)
	
    def _last_non_zero(self): #OPTIMIZACION para no calcular mas alla de los elementos distintos de ceros !
	return self.jet.nonzero()[0][-1] #Ultimo coeficeinte distinto de zero
	
    def _first_non_zero(self): #OPTIMIZACION para no calcular mas alla de los elementos distintos de ceros !
	return self.jet.nonzero()[0][0] #Ultimo coeficeinte distinto de zero
    
    def deriv(self,n=1):
	#aux=[]*(self.order-n)
	if self.jet.nonzero()[0].size > 0:
	  aux = []
	  for k in range(n,self._last_non_zero()+1):#+1
	    auxin = 1.#np.int32(1)
	    for i in range(n):
	      auxin *= k-i
	    
	    aux.append(auxin*self.jet[k])
	  
	  return taylorDiff(aux,self.order)
	else:
	  return taylorDiff([0.],self.order)
    
    def sameOrd(self,otro):
      orden = max(self.order,otro.order)
      self.jet.resize(orden+1)
      otro.jet.resize(orden+1)
      self.order=orden
      otro.order=orden
      return orden
    
    def extendOrd(self, orden = 75):
      self.jet.resize(orden+1)
      self.order=orden
      return taylorDiff(self.jet,self.order)
    
    def __sub__(self, otro):
	try:
	  if self.order == otro.order:
	    return taylorDiff(self.jet - otro.jet,otro.order)
	  else:
	    orden = max(self.order,otro.order)
	    self.jet.resize(orden+1)
	    otro.jet.resize(orden+1)
	    self.order=orden
	    otro.order=orden
	    return taylorDiff(self.jet - otro.jet ,orden)
	except:		## ADVERTENCIA!!!!!!!
	    return self.__sub__(taylorDiff(otro)) ## (RESULETO)Error de signo si es "float - taylorDiff" pero bien con el reciproco!
	  
    def __rsub__(self, otro):
      if not(isinstance(otro,taylorDiff)):
	return taylorDiff(otro) - self
      else:
        return self - otro
        
    def __mul__(self, otro): #Optimizacion : Numero de iteraciones y mejora en el orden! (formula con truncamiento de numero de operaciones)
        """
        Derivación Multiplicación
        """
        try:
	  orden = self.sameOrd(otro)
	  aux=[]
	  for k in range(orden+1):
	    #aux1=self.jet
	    #aux2=otro.jet
	    aux.append( np.dot( self.jet[0:k+1] , otro.jet[0:k+1][::-1] ) )
	  return taylorDiff( aux , orden )
        except:
          return taylorDiff(self.jet * otro, self.order)
        
    def __rmul__(self, otro):
        return self * otro
    
    def contract(self):
      return self.jet.sum()
    
    #def zerosOrder(self,otro):
      #azero=np.where(self.jet == 0.)
      #bzero=np.where(otro.jet == 0.)
      #if azero[0][-1] >= bzero[0][-1]:
	#return False
      #else:
	#return True
    
    def __div__(self, otro):# Tambien optimizaciones respecto a calculos de mas
        #"""
        #División, complicado t/t taylor dividido por taylor, t/c ; taylor entre constante; c/t con rdiv
        #"""
        try:## t/t
	  denNonZeros = otro.jet.nonzero()[0] # L' Hopital la serie no existe si  denZeros[0] > numZeros[0]
	  numNonZeros = self.jet.nonzero()[0] # la funcion division diverge para x = x_0
	  if denNonZeros.size == 0 and numNonZeros.size == 0: #todos lo coefs taylor cero
	      #print 'Bucle: ', 1
	      raise ZeroDivisionError
	    
	  elif denNonZeros.size == 0 and numNonZeros.size >= 0: #todos en el denom son cero
	      #print 'Bucle: ', 2
	      raise ZeroDivisionError
	    
	  elif denNonZeros.size >= 0 and numNonZeros.size == 0: #todos en el numerador cero
	      #print 'Bucle: ', 3
	      return 0.0
	    
	  else:
	      #print 'Bucle: ', 4
	      ell = numNonZeros[0] 
	      enn = denNonZeros[0]
	      if ell < enn:
		raise ZeroDivisionError
	      else:
		if ell == enn == 0: #Formula del libro
		  #print 'Method 1'
		  aux = [self.jet[0]/otro.jet[0]]
		  orden = self.sameOrd(otro)
		  for k in range(1,orden+1):
		    auxin = 0.
		    for i in range(k):
		      auxin += aux[i]*otro.jet[k-i]
		      
		    aux.append(1./otro.jet[0]*(self.jet[k]-auxin))
		  
		  return taylorDiff(aux,orden)#OK
		  
		else:
		  #print 'Method 2' # Mi formula (sera?)
		  orden = self.sameOrd(otro)
		  aux = np.zeros(ell-enn).tolist()
		  aux.append( self.jet[ell] / otro.jet[enn] )
		  self.extendOrd(orden+2*enn)
		  h2 = otro.__mul__(otro)
		  for k in np.arange(ell-enn+1,orden+1):
		    aux1 = 0.
		    aux2 = 0.
		    for i in range(ell,k+enn+1):
		      aux1 += self.jet[i]*otro.jet[k+enn-i]*(k+2*enn-2*i)
		    
		    for i in range(2*enn+1,k+2*enn+1):
		      aux2 += h2.jet[i]*aux[k+2*enn-i]*(k+2*enn-i)
		    
		    aux.append((1./(k*h2.jet[2*enn]))*(aux1-aux2))
		    
		  return taylorDiff( aux , orden )
	  
	except:
	  if not(isinstance(otro,taylorDiff)) and otro == 0:
	    raise ZeroDivisionError
	  elif isinstance(otro,taylorDiff):
	    raise ZeroDivisionError
	  else:
	    return taylorDiff(self.jet / otro, self.order)


    def __rdiv__(self, otro):
        #"""
        #División revrsa para poder usar floats en el numerador
        #"""
        if not(isinstance(otro,taylorDiff)):
	  #print 'Camino 1'
	  otro = taylorDiff([otro],self.order)
	  #print otro
	  return taylorDiff.__div__(otro,self)
	else:
	  #print 'Camino 2'
	  return otro / self

    def __pow__(self, power, method = 1):
        ##'''
        ##Operacion potencia para jets.
        ##'''
        # Casos simples!
        NonZeros = self.jet.nonzero()[0][0] #Primer termino de la serie distinto de 0 (potencia factorizable)
        #print 'NonZeros',NonZeros
        if not(isinstance(power,taylorDiff)):
	  if isinstance(power,int):
	    if power == 0:
	      return taylorDiff(1.0)
	    else:
	      if method == 1:
		#print 'Method',method
		aux = self
		if power > self.order:
		  self.extendOrd(power+self.order)
		  
		for n in range(1,power):
		  aux *= self
		
		return aux
		
	      else:
		#print 'Method: ',method
		#aux = np.zeros(NonZeros*power).tolist()
		aux = []
		aux.append( self.jet[NonZeros]**power )
		#self.extendOrd(self.order*power)
		for a in range(1,self.order-NonZeros):
		  auxin = 0.
		  for i in range(a):
		    #print self.order,i,a,a+NonZeros-i
		    auxin += (power*(a+NonZeros)-i*(power+1.))*aux[i]*self.jet[a+NonZeros-i]
		  
		  aux.append( (1./(self.jet[NonZeros]*(a-power*NonZeros))) * auxin )
		  
		return taylorDiff(aux,self.order)
		
	  else: # OJO si es un polinomio de orden menor n, regresa el poly factorizado por x^(n*power), ventaja: power real
	    #print 'real'
	    aux = []
	    aux.append( self.jet[NonZeros]**power )
	    orden = self.order
	    #self.extendOrd(orden)
	    for a in range(1,orden-NonZeros+1):
	      auxin = 0.
	      for i in range(a):
		#print self.order,i,a,a+NonZeros-i
		auxin += (power*(a+NonZeros)-i*(power+1.))*aux[i]*self.jet[a+NonZeros-i]
	      
	      aux.append( (1./(self.jet[NonZeros]*(a-power*NonZeros))) * auxin )
	      
	    return taylorDiff(aux,self.order)
		    
		
	else:
	  print 'FALTA DESARROLLO'
	  raise NotImplementedError
        #if not isinstance(power,taylorDiff) and self.jet[0] != 0:
	  #aux=[self.jet[0]**power] ## k=0 element
	  #for k in range(orden)[1:orden]:
	    #aux.append( (self.jet[k]-np.dot( np.array(aux)[0:k] , otro.jet[0:k][::-1] ))/otro.jet[0] )
	  
	  #return
	#elif:
	  
	  #return taylorDiff( aux , orden )
	  
        #return DifAuto (self.valor**n, n*self.valor**(n-1)*self.deriv)
#######################################################################
    def exp(self):
        """
        Exponencial
        """
        aux = [np.exp(self.jet[0])]
        for k in range(1,self.order+1):
	  auxin = 0.
	  for i in range(1,k+1):
	    auxin += i*self.jet[i]*aux[k-i]
	  
	  aux.append((1./k)*auxin)
	  
        return taylorDiff(aux,self.order)
        
    #def log(self):
        #"""
        #Logaritmo
        #"""
        #return DifAuto(math.log(self.valor), self.deriv / self.valor )
    
    #def sin(self):
        #"""
        #Seno
        #"""
        #return DifAuto(math.sin(self.valor), math.cos(self.valor) * self.deriv)
            
    #def cos(self):
        #"""
        #Coseno
        #"""
        #return DifAuto(math.cos(self.valor), -math.sin(self.valor) * self.deriv)

    #def tan(self):
        #"""
        #Tangente
        #"""
        #return self.sin()/self.cos()
            
        
#def exp(x):
    #try:
        #return x.exp()
    #except:
        #return math.exp(x)    
    
#def log(x):
    #try:
        #return x.log()
    #except:
        #return math.log(x)

#def cos(x):
    #try:
        #return x.cos()
    #except:
        #return math.cos(x)

#def sin(x):
    #try:
        #return x.sin()
    #except:
        #return math.sin(x)

#def tan(x):
    #try:
        #return x.tan()
    #except:
        #return math.tan(x)

