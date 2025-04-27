import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Nastavení pro lepší vizualizaci
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style('whitegrid')

# Načtení dat
print("Načítání dat...")
df = pd.read_csv('noisy_tokamak_data.csv')

# Základní průzkum dat
print("\nZákladní informace o datech:")
print(df.info())

print("\nStatistický přehled dat:")
print(df.describe())

print("\nPrvních 5 řádků dat:")
print(df.head())

# Kontrola chybějících hodnot
print("\nChybějící hodnoty podle sloupců:")
missing_values = df.isnull().sum()
print(missing_values)

# Vizualizace chybějících hodnot
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Vizualizace chybějících hodnot v datové sadě')
plt.tight_layout()
plt.show()

# ------ 1. ČIŠTĚNÍ DAT POMOCÍ TRADIČNÍCH METOD ------

# Kopie originálních dat pro srovnání metod
df_original = df.copy()
df_traditional = df.copy()

# A. Základní statistické metody pro detekci outlierů
print("\n--- Detekce outlierů pomocí statistických metod ---")

# Z-score metoda pro detekci outlierů
def detect_outliers_zscore(df, column, threshold=3):
    """Detekuje outliery pomocí Z-score metody"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if column in numeric_columns:
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        return df[abs(z_scores) > threshold].index
    return []

# IQR metoda pro detekci outlierů
def detect_outliers_iqr(df, column, k=1.5):
    """Detekuje outliery pomocí IQR metody"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if column in numeric_columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        return df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
    return []

# Aplikace detekce outlierů na numerické sloupce
outliers_by_column = {}
for column in df_traditional.select_dtypes(include=[np.number]).columns:
    if column != 'cas' and column != 'stabilita':  # Ignorujeme časový index a cílovou proměnnou
        z_score_outliers = detect_outliers_zscore(df_traditional, column)
        iqr_outliers = detect_outliers_iqr(df_traditional, column)
        
        # Kombinace outlierů detekovaných oběma metodami
        outliers = list(set(z_score_outliers) | set(iqr_outliers))
        outliers_by_column[column] = outliers
        
        print(f"Sloupec '{column}': Nalezeno {len(outliers)} outlierů")

# Vizualizace outlierů
plt.figure(figsize=(15, 10))
for i, column in enumerate(outliers_by_column.keys()):
    plt.subplot(3, 2, i+1)
    plt.scatter(df_traditional.index, df_traditional[column], alpha=0.5, label='Normální hodnoty')
    
    # Zvýraznění outlierů
    if outliers_by_column[column]:
        plt.scatter(outliers_by_column[column], 
                    df_traditional.loc[outliers_by_column[column], column], 
                    color='red', label='Outliery')
    
    plt.title(f'Outliery v {column}')
    plt.xlabel('Index měření')
    plt.ylabel(column)
    plt.legend()
plt.tight_layout()
plt.show()

# Fyzikální omezení a ošetření hodnot
print("\n--- Aplikace fyzikálních omezení ---")

# Definice fyzikálně možných rozsahů
physical_limits = {
    'teplota': (5, 30),     # keV - typický rozsah v tokamaku
    'proud': (5e5, 2e6),    # A - ampéry
    'magnet_pole': (1.5, 4), # T - tesla
    'hustota': (5e19, 2e20), # m^-3 - částice na metr krychlový
    'napeti': (2, 5),       # V - volty
    'beta_param': (0.1, 1.5)  # bezrozměrná veličina
}

# Aplikace fyzikálních omezení
for column, (lower, upper) in physical_limits.items():
    if column in df_traditional.columns:
        invalid_values = df_traditional[(df_traditional[column] < lower) | 
                                       (df_traditional[column] > upper)].index
        print(f"Sloupec '{column}': Nalezeno {len(invalid_values)} fyzikálně nemožných hodnot")
        
        # Nahrazení nemožných hodnot NaN
        df_traditional.loc[invalid_values, column] = np.nan

# Imputace chybějících hodnot
print("\n--- Imputace chybějících hodnot ---")

# Pro teplotu - lineární interpolace (předpokládá spojitý vývoj)
df_traditional['teplota'] = df_traditional['teplota'].interpolate(method='linear')

# Pro ostatní parametry - KNN imputace
columns_to_impute = [col for col in physical_limits.keys() if col != 'teplota']
imputer = KNNImputer(n_neighbors=5)
df_traditional[columns_to_impute] = imputer.fit_transform(df_traditional[columns_to_impute])

# ------ 2. ČIŠTĚNÍ DAT POMOCÍ AI METOD ------

# Kopie originálních dat pro AI metody
df_ai = df.copy()

print("\n--- Detekce outlierů pomocí Isolation Forest ---")

# Výběr pouze numerických sloupců pro Isolation Forest
numeric_columns = [col for col in df_ai.select_dtypes(include=[np.number]).columns 
                 if col != 'cas' and col != 'stabilita']

# Standardizace dat pro Isolation Forest
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_ai[numeric_columns]),
    columns=numeric_columns
)

# Použití Isolation Forest pro detekci outlierů
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_predictions = iso_forest.fit_predict(df_scaled)

# Identifikace outlierů (-1 označuje outlier)
outlier_indices = np.where(outlier_predictions == -1)[0]
print(f"Isolation Forest detekoval {len(outlier_indices)} outlierů")

# Vizualizace výsledků Isolation Forest v 2D prostoru pomocí PCA
pca = PCA(n_components=2)
df_pca = pd.DataFrame(
    pca.fit_transform(df_scaled),
    columns=['PC1', 'PC2']
)

plt.figure(figsize=(10, 8))
plt.scatter(df_pca.iloc[:, 0], df_pca.iloc[:, 1], 
           c=['red' if pred == -1 else 'blue' for pred in outlier_predictions],
           alpha=0.7)
plt.title('Vizualizace outlierů pomocí Isolation Forest a PCA')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm'), 
            label='Modrá=normální, Červená=outlier')
plt.grid(True)
plt.show()

# Použití výsledků Isolation Forest - označení outlierů v původních datech
df_ai.loc[outlier_indices, numeric_columns] = np.nan

print("\n--- Imputace chybějících hodnot pomocí pokročilých metod ---")

# Imputace chybějících hodnot pomocí fyzikálně informovaného modelu
# Příklad jednoduchého fyzikálně informovaného modelu:
# Teplota plazmatu je částečně závislá na proudu, magnetickém poli a beta parametru

# Zjednodušený fyzikální model pro teplotu:
# 1. Vybereme řádky, kde známe všechny relevantní parametry
valid_rows = df_ai.dropna(subset=['proud', 'magnet_pole', 'beta_param']).index

# 2. Natrénujeme lineární model na známých datech
X_train = df_ai.loc[valid_rows, ['proud', 'magnet_pole', 'beta_param']]
y_train = df_ai.loc[valid_rows, 'teplota']

# Vytvoření pipeline s normalizací a regresí
temp_model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
temp_model.fit(X_train, y_train)

# 3. Predikce chybějících hodnot teploty
missing_temp_rows = df_ai[df_ai['teplota'].isna()].index
if len(missing_temp_rows) > 0:
    X_missing = df_ai.loc[missing_temp_rows, ['proud', 'magnet_pole', 'beta_param']]
    df_ai.loc[missing_temp_rows, 'teplota'] = temp_model.predict(X_missing)
    print(f"Imputováno {len(missing_temp_rows)} chybějících hodnot teploty pomocí fyzikálního modelu")

# Použití DBSCAN pro imputaci ostatních parametrů
# DBSCAN je klastrovací algoritmus, který může pomoci identifikovat podobné vzorky
# a použít je pro imputaci

print("\n--- Imputace zbývajících chybějících hodnot pomocí DBSCAN a KNN ---")

# Příprava dat pro DBSCAN
df_complete_rows = df_ai.dropna()
if len(df_complete_rows) >= 5:  # Potřebujeme dostatek úplných řádků
    # Normalizace dat pro DBSCAN
    scaler_dbscan = StandardScaler()
    df_complete_scaled = scaler_dbscan.fit_transform(df_complete_rows[numeric_columns])
    
    # Použití DBSCAN pro klastrizaci
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    clusters = dbscan.fit_predict(df_complete_scaled)
    
    # Přidání informace o clusteru
    df_complete_rows['cluster'] = clusters
    
    # Pro každý řádek s chybějícími hodnotami najdeme nejvhodnější cluster
    for idx in df_ai.index:
        if idx not in df_complete_rows.index:  # Řádek s chybějícími hodnotami
            # Zjistíme, které sloupce mají hodnoty a které chybí
            row = df_ai.loc[idx]
            missing_cols = row[row.isna()].index.tolist()
            missing_cols = [c for c in missing_cols if c in numeric_columns]
            
            if len(missing_cols) > 0 and not all(row[numeric_columns].isna()):
                # Najdeme nejbližší cluster pomocí dostupných hodnot
                available_cols = [c for c in numeric_columns if c not in missing_cols]
                
                if available_cols:
                    # Normalizace dostupných hodnot
                    row_values = row[available_cols].values.reshape(1, -1)
                    available_indices = [numeric_columns.index(c) for c in available_cols]
                    
                    # Výpočet vzdálenosti k centrům clusterů
                    min_dist = float('inf')
                    best_cluster = -1
                    
                    for cluster_id in set(clusters):
                        if cluster_id != -1:  # Ignorujeme šumové body (-1)
                            cluster_points = df_complete_scaled[clusters == cluster_id]
                            if len(cluster_points) > 0:
                                # Výpočet průměrného bodu clusteru
                                cluster_center = np.mean(cluster_points, axis=0)
                                # Výpočet vzdálenosti pouze pomocí dostupných sloupců
                                dist = np.sqrt(np.sum((row_values - cluster_center[available_indices].reshape(1, -1))**2))
                                
                                if dist < min_dist:
                                    min_dist = dist
                                    best_cluster = cluster_id
                    
                    # Imputace chybějících hodnot pomocí průměru clusteru
                    if best_cluster != -1:
                        cluster_data = df_complete_rows[df_complete_rows['cluster'] == best_cluster]
                        for col in missing_cols:
                            df_ai.loc[idx, col] = cluster_data[col].mean()

# Použití KNN imputace pro zbývající chybějící hodnoty
imputer_ai = KNNImputer(n_neighbors=3)
df_ai[numeric_columns] = imputer_ai.fit_transform(df_ai[numeric_columns])

# ------ 3. SROVNÁNÍ VÝSLEDKŮ ČIŠTĚNÍ DAT ------

print("\n--- Srovnání výsledků tradičních a AI metod čištění dat ---")

# Vizualizace srovnání dat před a po čištění
plt.figure(figsize=(15, 10))
for i, column in enumerate(numeric_columns[:6]):  # Zobrazíme maximálně 6 parametrů
    plt.subplot(3, 2, i+1)
    
    # Originální data
    plt.scatter(df_original.index, df_original[column], 
               alpha=0.3, label='Originální data', color='blue')
    
    # Data po tradičním čištění
    plt.scatter(df_traditional.index, df_traditional[column], 
               alpha=0.5, label='Tradiční čištění', color='green')
    
    # Data po AI čištění
    plt.scatter(df_ai.index, df_ai[column], 
               alpha=0.7, label='AI čištění', color='red')
    
    plt.title(f'Srovnání metod čištění - {column}')
    plt.xlabel('Index měření')
    plt.ylabel(column)
    plt.legend()

plt.tight_layout()
plt.show()

# Korelační matice po čištění dat AI metodami
plt.figure(figsize=(10, 8))
correlation = df_ai[numeric_columns].corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
           vmin=-1, vmax=1)
plt.title('Korelační matice parametrů plazmatu po čištění dat AI metodami')
plt.tight_layout()
plt.show()

# Statistické srovnání čistých dat
print("\nStatistiky původních dat:")
print(df_original[numeric_columns].describe())

print("\nStatistiky po tradičním čištění:")
print(df_traditional[numeric_columns].describe())

print("\nStatistiky po AI čištění:")
print(df_ai[numeric_columns].describe())

# Srovnání konzistence dat z hlediska fyzikálních vztahů

# Příklad: V tokamaku by měl existovat vztah mezi teplotou a beta parametrem
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(df_traditional['teplota'], df_traditional['beta_param'], 
           alpha=0.7, label='Tradiční čištění')
plt.title('Vztah teplota-beta po tradičním čištění')
plt.xlabel('Teplota [keV]')
plt.ylabel('Beta parametr')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(df_ai['teplota'], df_ai['beta_param'], 
           alpha=0.7, label='AI čištění')
plt.title('Vztah teplota-beta po AI čištění')
plt.xlabel('Teplota [keV]')
plt.ylabel('Beta parametr')
plt.grid(True)

plt.tight_layout()
plt.show()

# Uložení vyčištěných dat
df_traditional.to_csv('tokamak_data_traditional_cleaned.csv', index=False)
df_ai.to_csv('tokamak_data_ai_cleaned.csv', index=False)

print("\nVyčištěná data byla uložena do souborů:")
print("- tokamak_data_traditional_cleaned.csv (tradiční metody)")
print("- tokamak_data_ai_cleaned.csv (AI metody)")

# ------ 4. FYZIKÁLNĚ INFORMOVANÁ VALIDACE DAT ------

print("\n--- Fyzikálně informovaná validace dat ---")

# Konečná validace pomocí známých fyzikálních vztahů
# Příklad: Vztah mezi beta parametrem, magnetickým polem a teplotou
# Beta parametr je úměrný poměru tlaku plazmatu a magnetického tlaku
# Zjednodušený vztah: beta ~ teplota * hustota / (magnet_pole^2)

# Výpočet očekávaného beta parametru
df_ai['expected_beta'] = df_ai['teplota'] * df_ai['hustota'] / (1e20 * df_ai['magnet_pole']**2)
df_ai['expected_beta'] = df_ai['expected_beta'] / df_ai['expected_beta'].mean() * df_ai['beta_param'].mean()

# Výpočet odchylky od očekávaného beta parametru
df_ai['beta_deviation'] = np.abs(df_ai['beta_param'] - df_ai['expected_beta']) / df_ai['expected_beta']

# Vizualizace srovnání skutečného a očekávaného beta parametru
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(df_ai.index, df_ai['beta_param'], label='Naměřený beta', color='blue', alpha=0.7)
plt.scatter(df_ai.index, df_ai['expected_beta'], label='Fyzikálně očekávaný beta', color='red', alpha=0.7)
plt.xlabel('Index měření')
plt.ylabel('Beta parametr')
plt.title('Srovnání naměřeného a fyzikálně očekávaného beta parametru')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(df_ai.index, df_ai['beta_deviation'], color='purple', alpha=0.7)
plt.axhline(y=0.2, color='red', linestyle='--', label='20% odchylka')
plt.xlabel('Index měření')
plt.ylabel('Relativní odchylka')
plt.title('Odchylka naměřeného beta od fyzikálně očekávaného')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Identifikace fyzikálně nekonzistentních hodnot
inconsistent_indices = df_ai[df_ai['beta_deviation'] > 0.2].index
print(f"Nalezeno {len(inconsistent_indices)} fyzikálně nekonzistentních měření (>20% odchylka)")

if len(inconsistent_indices) > 0:
    print("\nNekonzistentní měření:")
    print(df_ai.loc[inconsistent_indices, ['teplota', 'magnet_pole', 'hustota', 'beta_param', 'expected_beta', 'beta_deviation']])

# Finální úprava dat - korekce nekonzistentních hodnot
# Použití vážené hodnoty mezi naměřenou a očekávanou hodnotou
# Váha závisí na velikosti odchylky - větší odchylka = větší důvěra v fyzikální model
for idx in inconsistent_indices:
    deviation = df_ai.loc[idx, 'beta_deviation']
    weight = min(deviation, 0.8)  # Maximální váha modelu je 80%
    
    # Vážený průměr mezi naměřenou a očekávanou hodnotou
    corrected_value = (1 - weight) * df_ai.loc[idx, 'beta_param'] + weight * df_ai.loc[idx, 'expected_beta']
    df_ai.loc[idx, 'beta_param'] = corrected_value

print("\n--- Finální analýza stability dat ---")

# Kontrola, zda jsou data po čištění fyzikálně konzistentní
# Přepočítáme očekávaný beta parametr po korekci
df_ai['expected_beta'] = df_ai['teplota'] * df_ai['hustota'] / (1e20 * df_ai['magnet_pole']**2)
df_ai['expected_beta'] = df_ai['expected_beta'] / df_ai['expected_beta'].mean() * df_ai['beta_param'].mean()
df_ai['beta_deviation'] = np.abs(df_ai['beta_param'] - df_ai['expected_beta']) / df_ai['expected_beta']

# Průměrná odchylka od fyzikálního modelu
mean_deviation = df_ai['beta_deviation'].mean()
print(f"Průměrná odchylka od fyzikálního modelu po korekci: {mean_deviation:.3f} (ideálně < 0.1)")

# Kontrola stability pomocí modelu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Příprava dat pro model stability
X = df_ai[['teplota', 'proud', 'magnet_pole', 'hustota', 'napeti', 'beta_param']]
y = df_ai['stabilita']

# Vyhodnocení stability modelu pomocí cross-validace
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=5)

print(f"\nPřesnost modelu stability plazmatu na vyčištěných datech: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Důležitost parametrů pro predikci stability
rf_model.fit(X, y)
feature_importance = rf_model.feature_importances_

# Vizualizace důležitosti parametrů
plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.xlabel('Důležitost parametru')
plt.title('Důležitost parametrů pro predikci stability plazmatu')
plt.tight_layout()
plt.show()

print("\nZávěr: AI metody čištění dat v kombinaci s fyzikálně informovanou validací")
print("poskytují konzistentní dataset vhodný pro další analýzu a modelování")
print("stability plazmatu v tokamaku.")