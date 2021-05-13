COVID-CT klasifikator
Pozdravljeni, na hitro o modelu. Ideja klasifikacije je, da se vse rezine pacienta segmentirajo z uporabo Inf-Net-a (DengPing et al.),
s čimer se dobi količino anomalij na vsaki rezini. To naredimo za vsako rezino in izračunamo, kolikšen odsotek pljuč zajemajo bodisi
GGO, bodisi konsolidacije (ti dve maski na koncu izvrže InfNet). V prvem delu tekmovanja sva s pomočjo statistike razdelila paciente na 1 % velike intervale
in izračunala verjetnost, da ima pacient, katerega odstotek anomalije pljuč paše v interval, obsežno COVID pljučnico. Na koncu še savgol filter za izravnavo in to je to. Statistična obdelava se nahaja v Stats.ipynb - začetni del sodi v drugi del tekmovanja, končni del pa je bil del prvega dela tekmovanja.
V drugem delu tekmovanja sva spet izračunala odstotek pljuč, ki je patološki in z uporabo funkcije calculate_thresholds() v stats.ipynb določila meje za prestop med A in B ter B in C, tako da se maksimizira območje znotraj intervala - da pripada čimveč ploščine tistemu razredu.
Model Inf-Net in postopek pridobitve mask ter izračun skupnih odstotkov se nahaja v Do.ipynb. Glavna funkcija, ki opravlja večino dela je process_one_patient(). Predprocesiranje je zelo podobno kot v vzorčnem primeru, le da nisva uporabila samo centralnih rezin, ampak vse.
white_check_mark
eyes
raised_hands





8:35
Inf-Net model je bil zelo kompleksen za pripravo, zato bo mogoče zgledalo malce zapleteno. V osnovi gre pa tako. Najprej se iz posamezne rezine generira semi-inf -net maska, ki je bolj groba in ne diskriminira med razredi. Potem se original rezina in ta nova maska "concatenate"-jo in pošljejo v drug model, ki uporablja UNet. Osnova obeh modelov pa je ResNet. (edited) 
8:36
Žal v tem času nisva uspela dobro dokumentirati kode, ker nisva vedela, kaj naju čaka v drugem krogu, zato prosim, da če imate kakšno vprašanje glede implementacije, naju kontaktirate.
8:37
Lep pozdrav, Filip in Tim
