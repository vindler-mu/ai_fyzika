VEŠKERÝ OBSAH TÉTO SLOŽKY BYL GENEROVÁN MODELEM CLAUDE SONET 3.5

## ČIŠTĚNÍ

Vytvořil jsem pro vás komplexní ukázku dat a kódu pro čištění fyzikálních dat pomocí AI technik. Poskytnuté materiály zahrnují:

1.  **Syntetická data tokamaku s problémy** (`noisy_tokamak_data.csv`):
    -   Chybějící hodnoty (v některých sloupcích chybí data)
    -   Odlehlé hodnoty (extrémní, nefyzikální hodnoty jako -15.48 nebo 97.23)
    -   Nekonzistentní data (hodnoty, které porušují fyzikální vztahy)
2.  **Komplexní skript pro čištění dat**, který demonstruje:
    -   Tradiční metody čištění dat
    -   AI metody čištění dat
    -   Srovnání obou přístupů
    -   Fyzikálně informovanou validaci dat

### Hlavní AI techniky pro čištění dat v ukázce:

1.  **Isolation Forest pro detekci anomálií**
    -   Detekuje odlehlé hodnoty v mnohorozměrném prostoru
    -   Na rozdíl od tradičních metod bere v úvahu vztahy mezi parametry
2.  **DBSCAN pro clusterovou imputaci**
    -   Klastrovací algoritmus, který pomáhá najít podobné záznamy
    -   Imputuje chybějící hodnoty na základě podobných měření
3.  **Fyzikálně informovaný model pro imputaci**
    -   Jednoduchý model založený na fyzikálních vztazích mezi parametry
    -   Využívá vztah mezi teplotou a ostatními parametry
4.  **Validace dat pomocí fyzikálních zákonů**
    -   Kontroluje, zda vyčištěná data odpovídají známým fyzikálním vztahům
    -   Automaticky opravuje hodnoty, které porušují fyzikální zákony

### Vizualizace a srovnání:

Kód obsahuje množství vizualizací, které ukazují:

-   Detekci outlierů tradičními metodami vs. Isolation Forest
-   Srovnání dat před a po čištění
-   Ověření fyzikální konzistence dat pomocí vztahu mezi parametry
-   Vyhodnocení důležitosti parametrů pro predikci stability

### Výhody AI přístupu k čištění dat:

1.  **Automatická detekce komplexních vzorů** - AI metody rozpoznají vícerozměrné anomálie, které tradiční metody přehlíží
2.  **Zachování fyzikálních vztahů** - Fyzikálně informované modely zajišťují konzistenci dat
3.  **Lepší imputace chybějících hodnot** - AI bere v úvahu všechny dostupné informace pro přesnější odhady
4.  **Škálovatelnost** - AI metody lze snadno aplikovat na velké množství parametrů a měření

Tento kód můžete použít jako výchozí bod pro vlastní čištění fyzikálních dat nebo jej rozšířit o další AI techniky podle potřeby.


## VIZUALIZACE

### Co skript obsahuje:

-   **Modulární struktura** - skript je rozdělen do logických funkcí pro jednotlivé typy vizualizací
-   **Hlavní funkce `main()`** - orchestruje postupné spouštění všech analýz
-   **Dokumentace všech funkcí** - každá funkce obsahuje dokumentační string popisující její účel
-   **Progresivní postup analýz** - od základních vizualizací k pokročilým AI metodám

### Hlavní komponenty AI vizualizace dat:

1.  **Pokročilé časové vizualizace**
    -   Vizualizace parametrů plazmatu v čase obohacená o detekci režimů
    -   Distribuce hodnot v různých režimech plazmatu
2.  **Analýza korelací s AI asistencí**
    -   Clusterovaná korelační matice pro identifikaci souvisejících parametrů
    -   Určení důležitosti parametrů pro stabilitu pomocí Random Forest
3.  **Dimenzionální redukce pro vizualizaci vícerozměrných dat**
    -   PCA (Principal Component Analysis) - základní lineární redukce
    -   t-SNE - zachování lokální struktury dat
    -   UMAP - nejmodernější metoda kombinující výhody globální i lokální projekce
    -   Kernel PCA - nelineární varianta PCA pro zachycení komplexních vztahů
4.  **Shlukovací analýza pro identifikaci režimů plazmatu**
    -   K-means clustering - klasická metoda pro shlukování
    -   DBSCAN - density-based clustering odolný vůči šumu
    -   Porovnání AI-objevených clusterů s fyzikálně definovanými režimy
5.  **3D vizualizace fázového prostoru plazmatu**
    -   Interaktivní 3D graf s fyzikálně významnými parametry
    -   Barevné kódování podle režimů
6.  **Analýza stabilních oblastí v parametrickém prostoru**
    -   Mapa stability v prostoru nejdůležitějších parametrů
    -   Pravděpodobnostní vizualizace stability
7.  **Animovaná trajektorie plazmatu**
    -   Dynamická vizualizace pohybu plazmatu v redukovaném prostoru parametrů
8.  **Vizualizace vztahů mezi všemi parametry**
    -   Pair plot s barevným kódováním podle režimů
9.  **Vizualizace predikce stability**
    -   Confusion matrix, ROC křivka, Precision-Recall křivka
    -   Distribuce pravděpodobností stability

Skript je navržený tak, aby byl intuitivní a mohli jste ho upravovat pro vlastní datové sady. Například můžete snadno změnit vstupní datový soubor v hlavní funkci nebo upravit parametry jednotlivých vizualizací podle vlastních preferencí.

Pro použití tohoto skriptu s vlastními fyzikálními daty stačí upravit strukturu CSV souboru tak, aby obsahovala podobné parametry, nebo upravit jména sloupců v kódu podle vašich vlastních dat.
