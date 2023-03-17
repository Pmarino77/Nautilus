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


#numeros ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10''11', '12', '13', '14', '15.16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50']
#Letras
#letra_minus=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'ñ', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'z']
#letra_mayus=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Z']
#simbolos=['!', '#', '$', '%', '&', '/', '(', ')', '*']

