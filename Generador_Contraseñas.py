#Generador de contraseñas
input("¡Bienvenido al programa para generar contraseñas aleatorias de la SIPyV! Presiona ENTER para continuar")
n=int(input("¿Cuántos caracteres deseas que tenga tu contraseña?: "))
import random, string

def creaPass(n):
    todas=list(string.ascii_letters)+list(string.digits)+list(string.punctuation)
    passw=[]
    for i in range(n):
        tpm=random.choice(todas)
        passw.append(tpm)
    res="".join(passw)
    return res

test=creaPass(n)
print(test)


