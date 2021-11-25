# Regular expressions
import re

def fun1(): 
    str = "Laba diena sese kylas, kazkas vakarais velai lol iki.."
    str2 = "vakar lol"

    #Kurioje sakinio vietoe prasideda 'vak'
    pos = -1

    for i in range(2, len(str2)):
        if (str2[i] == 'k' and str2[i-1] == 'a' and str2[i-2] == 'v'):
            pos = i - 2

    print(str2[0])
    print(pos)

pattern = '00370'
str3 = '003780000000380'

if re.match(pattern,str3):
    print('atitinka')
else:
    print('nelabai')

print('----------------- specialus imboliai----------')
# . - atstoja bet kuri simboli
# search - atlieka pasieska pagal pozimui
print(re.search(r'Saus.ini.','Sausuinis mazas 5141').group())

# ^ - sulyginti zodzius tik teksto pradzioje, pirma zodi
print(re.search(r'^Eat', 'Eat cake...').group())
print(re.search(r'^eat', 'Lets eat cake...'))

# $ - sulyginti zodzius sakinio gale, paskutini zodi
print(re.search(r'cake$', 'Jonai ar skanus cake').group())
print(re.search(r'cake$', 'Jonai ar skanus naminis'))


# [abc] - ieskoti tik mazu a, b,x raidziu
# [a-zA-Z0-9] - yra ieskomis visos lotyniskos ir skaitmenys
print(re.search(r'[0-6]','Mano laimingas skaicius yra: 4').group())
print(re.search(r'[0-6]','Mano laimingas skaicius yra: 7'))
print(re.search(r'Number: [0-6]','Number: 2').group())
# print(re.search(r'Number:.[^2-6]','Number: 2234').group())


# \ - blacklash
print(re.search(r'kaskas\sJonas\sPetras', 'kaskas Jonas Petras').group())

# \w - atitinka iskyrus spec simbolius
# \W - atitinka viska kas nepaluola po \w
# \ s tarpys tabus new line
# \S simboliai kurie nepapuola po \s
print('Lower case: ', re.search('Co\wk\we', 'Cookie').group())
print('Upper case: ', re.search('Co\Wk\We', 'Co@k@e').group())

# \d palygina skaitmena
# \D - palygima viskas kas nepaluoja po \d

print('Kiek turi sausainiu: ', re.search(r'\d','As turiu 1045445 sausainiu').group())
print('Kiek turi sausainiu: ', re.search(r'\d+','As turiu 1045445 sausainiu').group())
print('Kiek turi sausainiu: ', re.search(r'\d+\D\d','As turiu 1045445.5 sausainiu').group())



skaicius = float( re.search(r'\d+\D\d','As turiu 104.5 sausainiu').group()) * 2234

print(skaicius)

# \t - palygina tabus
# \n - newline
# \r - return
# \A - ^ lygina simboli zodzio pradzioje
# \Z - zodzio gale palygina simboli
# \b -

#suranda is eoles atsikartojanto simboli
print(re.search(r'Co+kie', 'Cooooooooooooooooooookie').group())

# * - patikrina ar nurodytas simbolis atsikartoja viena ir daugiau kartu
print(re.search(r'Co*a*kie', 'Cooooooooooooooooooaaaaaaaaaaakie').group())
# ? tiksliai nuli arba viena karta
print(re.search(r'Colo?r', 'Color').group())


# {x} x yra skaicius kiek reikia atlikti paieska
# {x,} x kartu ir daugiau
# {x,y} x kartu ir daugiau bet ne daugiau nei y kartu
print(re.search(r'\d{4}', 'skaicius 4444sda').group())
print(re.search(r'\d{4,}', 'skaicius 444445566645').group())
print(re.search(r'\d{4,8}', 'skaicius 444445566645').group())
print(re.search(r'[a-z]{4,8}', 'skaicius kadszzfdkazdzzz').group())

# () - reg exp grupavimui
tekstas = 'Prasau susisiekti su bendrabucio adminu el pastu jonas.jonaitis@ktu.edu del suduzusio tuoleto'
match = re.search(r'([\w\.-]+)@([\w\.-]+)',tekstas)

print(match)

print('el pastas:', match.group())
print('vartotojo vardas:', match.group(1))
print('hostas :', match.group(2))

# (?P<name>)
match2 = re.search(r'(?P<email>(?P<username>[\w\.-]+)@(?P<hostas>[\w\.-]+))',tekstas)
print("el pastas: ", match2.group('email'))
print("vartotojo vardas: ", match2.group('username'))
print("hostas: ", match2.group('hostas'))


# findall
# () - reg exp grupavimui
tekstas = 'Prasau susisiekti su bendrabucio adminu el pastu petras@yaho.com jonas.jonaitis@ktu.edu; kristijonas@gmail.com del suduzusio tuoleto'
email = re.findall(r'[\w\.-]+@[\w\.-]+',tekstas)

print(email)

for i in range(0, len(email)):
    print(email[i])

tekstas = 'Admionistradtoriu numeriai: 8684545 arba 867812213'
numeriai = re.findall(r'\d+',tekstas)
print(numeriai)



