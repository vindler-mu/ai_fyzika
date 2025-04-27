import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, MDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from umap import UMAP
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

# Nastavení pro lepší vizualizaci
plt.rcParams['figure.figsize'] = (14, 10)
plt.style.use('seaborn-v0_8-whitegrid')
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="whitegrid", rc=custom_params)

# Vlastní barevné palety
plasma_cmap = LinearSegmentedColormap.from_list("plasma_modes", 
                                             [(0, '#2c3e50'), (0.33, '#3498db'), 
                                              (0.66, '#f39c12'), (1, '#e74c3c')])

def load_data(filepath):
    """Načtení dat z tokamaku."""
    print(f"Načítání dat tokamaku z {filepath}...")
    df = pd.read_csv(filepath)
    
    # Základní informace o datech
    print(f"Počet měření: {len(df)}")
    print(f"Počet parametrů: {len(df.columns)}")
    print(f"Časový rozsah: {df['cas'].min()} - {df['cas'].max()}")
    
    return df

def create_time_series_visualizations(df):
    """Vytvoření pokročilých časových vizualizací."""
    print("\nVytváření pokročilých časových vizualizací...")
    
    plt.figure(figsize=(14, 10))
    plt.suptitle('Vývoj parametrů plazmatu v čase s režimy', fontsize=16, y=0.95)
    
    # Použijeme vlastní rozložení
    gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1])
    
    # 1. Teplota plazmatu
    ax1 = plt.subplot(gs[0, 0])
    sns.lineplot(x='cas', y='teplota', data=df, ax=ax1, linewidth=2)
    ax1.scatter(df['cas'], df['teplota'], c=df['rezim'].astype('category').cat.codes, 
               cmap=plasma_cmap, s=50, alpha=0.7)
    ax1.set_title('Teplota plazmatu [keV]')
    ax1.set_xlabel('')
    
    # 1b. Distribuce teploty
    ax1b = plt.subplot(gs[0, 1])
    sns.kdeplot(data=df, x='teplota', hue='rezim', fill=True, ax=ax1b, palette='viridis')
    ax1b.set_title('Distribuce teploty')
    ax1b.set_xlabel('Teplota [keV]')
    
    # 2. Beta parametr
    ax2 = plt.subplot(gs[1, 0])
    sns.lineplot(x='cas', y='beta_param', data=df, ax=ax2, linewidth=2)
    ax2.scatter(df['cas'], df['beta_param'], c=df['rezim'].astype('category').cat.codes, 
               cmap=plasma_cmap, s=50, alpha=0.7)
    ax2.set_title('Beta parametr')
    ax2.set_xlabel('')
    
    # 2b. Distribuce beta parametru
    ax2b = plt.subplot(gs[1, 1])
    sns.kdeplot(data=df, x='beta_param', hue='rezim', fill=True, ax=ax2b, palette='viridis')
    ax2b.set_title('Distribuce beta parametru')
    ax2b.set_xlabel('Beta parametr')
    
    # 3. Magnetické pole
    ax3 = plt.subplot(gs[2, 0])
    sns.lineplot(x='cas', y='magnet_pole', data=df, ax=ax3, linewidth=2)
    ax3.scatter(df['cas'], df['magnet_pole'], c=df['rezim'].astype('category').cat.codes, 
               cmap=plasma_cmap, s=50, alpha=0.7)
    ax3.set_title('Magnetické pole [T]')
    ax3.set_xlabel('Čas [s]')
    
    # 3b. Distribuce magnetického pole
    ax3b = plt.subplot(gs[2, 1])
    sns.kdeplot(data=df, x='magnet_pole', hue='rezim', fill=True, ax=ax3b, palette='viridis')
    ax3b.set_title('Distribuce magnetického pole')
    ax3b.set_xlabel('Magnetické pole [T]')
    
    # Legenda pro celý graf
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plasma_cmap(i/5), 
                                  label=mode, markersize=10) 
                      for i, mode in enumerate(df['rezim'].unique())]
                      
    plt.figlegend(handles=legend_elements, loc='lower center', ncol=len(df['rezim'].unique()), 
                 bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def analyze_correlations(df, numeric_columns):
    """Analýza korelací s AI asistovanou interpretací."""
    print("\nAnalýza korelací a vztahů mezi parametry...")
    
    # Korelační matice
    plt.figure(figsize=(12, 10))
    plt.suptitle('Analýza korelací parametrů plazmatu', fontsize=16)
    
    # Spočítáme korelace
    corr = df[numeric_columns + ['stabilita']].corr()
    
    # Vizualizace korelační matice s clusteringem
    sns.clustermap(corr, cmap='coolwarm', annot=True, fmt='.2f', center=0,
                  linewidths=0.5, figsize=(12, 10))
    plt.suptitle('Clusterovaná korelační matice - hierarchické shlukování podobných parametrů', 
                fontsize=16, y=0.95)
    plt.show()
    
    # Příprava dat pro Random Forest feature importance analýzu
    X_rf = df[numeric_columns]
    y_rf = df['stabilita']
    
    # Trénování Random Forest pro analýzu důležitosti parametrů
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_rf, y_rf)
    
    # Standardní feature importance
    importances = rf_model.feature_importances_
    
    # Permutační důležitost (robustnější metrika)
    result = permutation_importance(rf_model, X_rf, y_rf, n_repeats=10, random_state=42)
    perm_importances = result.importances_mean
    
    # Vizualizace důležitosti parametrů z Random Forest
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    feat_importances = pd.Series(importances, index=numeric_columns)
    feat_importances.sort_values().plot(kind='barh', colormap='viridis')
    plt.title('Důležitost parametrů (MDI)')
    plt.xlabel('Relativní důležitost')
    
    plt.subplot(1, 2, 2)
    perm_feat_importances = pd.Series(perm_importances, index=numeric_columns)
    perm_feat_importances.sort_values().plot(kind='barh', colormap='viridis')
    plt.title('Důležitost parametrů (Permutační)')
    plt.xlabel('Pokles přesnosti')
    
    plt.tight_layout()
    plt.show()
    
    return importances

def dimension_reduction_visualization(X_scaled, df):
    """Dimenzionální redukce pro vizualizaci vícerozměrných dat."""
    print("\nAplikace metod AI pro dimenzionální redukci a vizualizaci vícerozměrných dat...")
    
    # Připravíme barevné kódování podle režimu
    regime_colors = df['rezim'].astype('category').cat.codes
    
    # Vytvoříme multiplot s různými metodami dimenzionální redukce
    plt.figure(figsize=(18, 14))
    plt.suptitle('Vizualizace vícerozměrných dat plazmatu pomocí AI redukce dimenzí', fontsize=18)
    
    # 1. PCA (Principal Component Analysis)
    plt.subplot(2, 2, 1)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=regime_colors, cmap=plasma_cmap, s=50, alpha=0.7)
    plt.title(f'PCA projekce (vysvětlená variance: {pca.explained_variance_ratio_.sum():.2%})')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(label='Režim plazmatu')
    
    # 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
    plt.subplot(2, 2, 2)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(X_scaled)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=regime_colors, cmap=plasma_cmap, s=50, alpha=0.7)
    plt.title('t-SNE projekce (zachovává lokální strukturu)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(label='Režim plazmatu')
    
    # 3. UMAP (Uniform Manifold Approximation and Projection)
    plt.subplot(2, 2, 3)
    umap = UMAP(n_components=2, random_state=42)
    umap_result = umap.fit_transform(X_scaled)
    plt.scatter(umap_result[:, 0], umap_result[:, 1], c=regime_colors, cmap=plasma_cmap, s=50, alpha=0.7)
    plt.title('UMAP projekce (zachovává globální i lokální strukturu)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.colorbar(label='Režim plazmatu')
    
    # 4. Kernel PCA (nelineární PCA)
    plt.subplot(2, 2, 4)
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
    kpca_result = kpca.fit_transform(X_scaled)
    plt.scatter(kpca_result[:, 0], kpca_result[:, 1], c=regime_colors, cmap=plasma_cmap, s=50, alpha=0.7)
    plt.title('Kernel PCA projekce (nelineární vztahy)')
    plt.xlabel('KPCA 1')
    plt.ylabel('KPCA 2')
    plt.colorbar(label='Režim plazmatu')
    
    # Přidání legendy
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plasma_cmap(i/5), 
                                  label=mode, markersize=10) 
                      for i, mode in enumerate(df['rezim'].unique())]
                      
    plt.figlegend(handles=legend_elements, loc='lower center', ncol=len(df['rezim'].unique()), 
                 bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    
    return pca, pca_result

def clustering_analysis(X_scaled, pca_result, df):
    """Identifikace režimů plazmatu pomocí shlukovací analýzy."""
    print("\nIdentifikace režimů plazmatu pomocí shlukovací analýzy...")
    
    # Příprava dat pro clustering
    X_cluster = X_scaled
    
    # Určení optimálního počtu clusterů pomocí metody Elbow
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_cluster)
        inertia.append(kmeans.inertia_)
    
    # Vizualizace optimálního počtu clusterů
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'o-')
    plt.title('Elbow metoda pro určení optimálního počtu clusterů')
    plt.xlabel('Počet clusterů')
    plt.ylabel('Inertia (suma čtverců vzdáleností)')
    plt.grid(True)
    plt.show()
    
    # Aplikace K-means s optimálním počtem clusterů (například 5)
    n_clusters = 5  # Můžeme zvolit na základě Elbow grafu
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_cluster)
    
    # Aplikace DBSCAN (density-based clustering)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_cluster)
    
    # Vizualizace výsledků clusteringu ve 2D prostoru pomocí PCA
    plt.figure(figsize=(16, 8))
    plt.suptitle('Shlukování parametrů plazmatu pomocí AI', fontsize=16)
    
    # K-means clustering
    plt.subplot(1, 2, 1)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.7)
    plt.title('K-means clustering')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(label='Cluster')
    
    # DBSCAN clustering
    plt.subplot(1, 2, 2)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=dbscan_labels, cmap='viridis', s=50, alpha=0.7)
    plt.title('DBSCAN clustering (density-based)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(label='Cluster')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    # Srovnání AI-objevených clusterů s fyzikálními režimy
    plt.figure(figsize=(12, 5))
    plt.suptitle('Srovnání AI-objevených režimů s fyzikálními režimy plazmatu', fontsize=16)
    
    # Contingency matrix (kolik vzorků z každého fyzikálního režimu spadá do každého clusteru)
    contingency_table = pd.crosstab(df['rezim'], pd.Series(cluster_labels, name='AI Cluster'))
    
    # Vizualizace contingency matrix jako heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(contingency_table, annot=True, cmap="YlGnBu", fmt='d', cbar=False)
    plt.title('Porovnání fyzikálních režimů a AI clusterů')
    plt.xlabel('AI-objevený cluster')
    plt.ylabel('Fyzikální režim')
    
    # Vizualizace správnosti AI clusterů
    plt.subplot(1, 2, 2)
    # Normalizace po řádcích
    contingency_norm = contingency_table.div(contingency_table.sum(axis=1), axis=0)
    sns.heatmap(contingency_norm, annot=True, cmap="YlGnBu", fmt='.2f', cbar=True)
    plt.title('Poměr fyzikálních režimů v AI clusterech')
    plt.xlabel('AI-objevený cluster')
    plt.ylabel('Fyzikální režim')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    return cluster_labels, n_clusters, contingency_table

def create_3d_visualization(df):
    """Trojrozměrná vizualizace fázového prostoru plazmatu."""
    print("\nVytváření 3D vizualizace fázového prostoru plazmatu...")
    
    # Připravíme barevné kódování podle režimu
    regime_colors = df['rezim'].astype('category').cat.codes
    
    # Vytvoření 3D grafu s fyzikálně informovanými osami
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Vybereme tři fyzikálně významné parametry
    x = df['beta_param']
    y = df['teplota']
    z = df['magnet_pole']
    
    # Barevné kódování podle režimu
    scatter = ax.scatter(x, y, z, c=regime_colors, cmap=plasma_cmap, s=50, alpha=0.7)
    
    # Přidání popisu os
    ax.set_xlabel('Beta parametr')
    ax.set_ylabel('Teplota [keV]')
    ax.set_zlabel('Magnetické pole [T]')
    
    # Přidání barevné stupnice
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Režim plazmatu')
    
    # Přidání titulku
    plt.title('3D fázový prostor plazmatu')
    
    # Přidáme legendu
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plasma_cmap(i/5), 
                                  label=mode, markersize=10) 
                      for i, mode in enumerate(df['rezim'].unique())]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 0.8))
    
    # Nastavení dynamické rotace pro interaktivní prohlížení
    for angle in range(0, 360, 5):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)
    
    plt.show()

def analyze_stability_regions(df, numeric_columns, X_scaled, scaler):
    """Analýza stabilních oblastí v parametrickém prostoru."""
    print("\nAnalýza stabilních oblastí v parametrickém prostoru plazmatu...")
    
    # Připravíme data pro klasifikaci stability
    X_stability = X_scaled
    y_stability = df['stabilita']
    
    # Natrénujeme Random Forest pro predikci stability
    X_train, X_test, y_train, y_test = train_test_split(X_stability, y_stability, test_size=0.3, random_state=42)
    rf_stability = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_stability.fit(X_train, y_train)
    
    # Vytvoříme grid bodů pro dva nejvýznamnější parametry
    param1 = 'beta_param'  # Parametr s nejvyšší důležitostí
    param2 = 'teplota'     # Parametr s druhou nejvyšší důležitostí
    
    # Rozsahy parametrů
    param1_range = np.linspace(df[param1].min(), df[param1].max(), 100)
    param2_range = np.linspace(df[param2].min(), df[param2].max(), 100)
    param1_mesh, param2_mesh = np.meshgrid(param1_range, param2_range)
    
    # Připravíme data pro predikci
    mesh_points = np.c_[param1_mesh.ravel(), param2_mesh.ravel()]
    
    # Doplníme průměrné hodnoty ostatních parametrů
    other_params = [col for col in numeric_columns if col not in [param1, param2]]
    avg_values = df[other_params].mean().values
    mesh_data = np.zeros((mesh_points.shape[0], len(numeric_columns)))
    
    # Indexy parametrů v originálních datech
    param1_idx = numeric_columns.index(param1)
    param2_idx = numeric_columns.index(param2)
    
    # Naplníme mesh_data průměrnými hodnotami
    for i in range(len(numeric_columns)):
        if i == param1_idx:
            mesh_data[:, i] = mesh_points[:, 0]
        elif i == param2_idx:
            mesh_data[:, i] = mesh_points[:, 1]
        else:
            other_idx = other_params.index(numeric_columns[i])
            mesh_data[:, i] = avg_values[other_idx]
    
    # Standardizace dat
    mesh_data_scaled = scaler.transform(mesh_data)
    
    # Predikce stability pro každý bod mřížky
    stability_pred = rf_stability.predict(mesh_data_scaled)
    stability_pred_proba = rf_stability.predict_proba(mesh_data_scaled)[:, 1]  # Pravděpodobnost stability
    
    # Reshape výsledků pro contour plot
    stability_mesh = stability_pred.reshape(param1_mesh.shape)
    stability_proba_mesh = stability_pred_proba.reshape(param1_mesh.shape)
    
    # Vizualizace stabilních oblastí
    plt.figure(figsize=(18, 8))
    plt.suptitle('Analýza stabilních oblastí v parametrickém prostoru plazmatu', fontsize=16)
    
    # 1. Binární stabilita (stabilní/nestabilní)
    plt.subplot(1, 2, 1)
    contour = plt.contourf(param1_mesh, param2_mesh, stability_mesh, 
                          levels=[-0.5, 0.5, 1.5], colors=['#e74c3c', '#2ecc71'], alpha=0.7)
    plt.scatter(df[param1], df[param2], c=df['stabilita'], cmap='coolwarm', s=50, edgecolor='k')
    plt.colorbar(ticks=[0, 1], label='Stabilita')
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.title('Binární mapa stability')
    
    # 2. Pravděpodobnost stability
    plt.subplot(1, 2, 2)
    contour_proba = plt.contourf(param1_mesh, param2_mesh, stability_proba_mesh, 
                                levels=np.linspace(0, 1, 11), cmap='coolwarm', alpha=0.7)
    plt.scatter(df[param1], df[param2], c=df['stabilita'], cmap='coolwarm', s=50, edgecolor='k')
    plt.colorbar(label='Pravděpodobnost stability')
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.title('Pravděpodobnostní mapa stability')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def create_animated_trajectory(df, X_scaled):
    """Animovaná časová vizualizace trajektorie v prostoru parametrů."""
    print("\nVytváření animované vizualizace trajektorie plazmatu...")
    
    # Použijeme PCA pro redukci dimenzionality
    pca_traj = PCA(n_components=2)
    traj_data = pca_traj.fit_transform(X_scaled)
    
    # Připravíme barevné kódování podle režimu
    regime_colors = df['rezim'].astype('category').cat.codes
    
    # Vytvoření figury pro animaci
    fig_anim, ax_anim = plt.subplots(figsize=(10, 8))
    ax_anim.set_xlim(traj_data[:, 0].min() - 0.5, traj_data[:, 0].max() + 0.5)
    ax_anim.set_ylim(traj_data[:, 1].min() - 0.5, traj_data[:, 1].max() + 0.5)
    ax_anim.set_xlabel(f'PC1 ({pca_traj.explained_variance_ratio_[0]:.2%} variance)')
    ax_anim.set_ylabel(f'PC2 ({pca_traj.explained_variance_ratio_[1]:.2%} variance)')
    ax_anim.set_title('Trajektorie plazmatu v redukovaném prostoru parametrů')
    
    # Přidáme barevné body podle režimu pro celou trajektorii (pro referenci)
    scatter_bg = ax_anim.scatter(traj_data[:, 0], traj_data[:, 1], c=regime_colors, 
                               cmap=plasma_cmap, alpha=0.2, s=30)
    
    # Bod pro aktuální pozici
    point, = ax_anim.plot([], [], 'ro', markersize=10)
    
    # Linka pro trajektorii
    line, = ax_anim.plot([], [], 'b-', linewidth=2, alpha=0.7)
    
    # Textová anotace pro čas a parametry
    time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes, fontsize=12)
    param_text = ax_anim.text(0.02, 0.90, '', transform=ax_anim.transAxes, fontsize=10)
    
    # Inicializační funkce pro animaci
    def init():
        point.set_data([], [])
        line.set_data([], [])
        time_text.set_text('')
        param_text.set_text('')
        return point, line, time_text, param_text
    
    # Funkce pro aktualizaci animace v každém kroku
    def update(frame):
        x_data = traj_data[:frame+1, 0]
        y_data = traj_data[:frame+1, 1]
        
        point.set_data(x_data[-1], y_data[-1])
        line.set_data(x_data, y_data)
        
        time_text.set_text(f'Čas: {df["cas"].iloc[frame]:.1f} s')
        param_text.set_text(f'Teplota: {df["teplota"].iloc[frame]:.2f} keV, Beta: {df["beta_param"].iloc[frame]:.2f}')
        
        return point, line, time_text, param_text
    
    # Vytvoření animace
    anim = FuncAnimation(fig_anim, update, frames=len(df), init_func=init, 
                        interval=100, blit=True)
    
    # Uložení animace jako GIF (volitelné)
    # Poznámka: Pro uložení GIF je potřeba mít nainstalovaný 'imagemagick' nebo 'pillow'
    try:
        anim.save('plasma_trajectory.gif', writer='pillow', fps=10)
        print("Animace byla uložena jako 'plasma_trajectory.gif'")
    except Exception as e:
        print(f"Nepodařilo se uložit animaci: {e}")
        print("Pro zobrazení animace použijte plt.show()")
    
    plt.tight_layout()
    plt.show()

def create_pair_plot(df, numeric_columns):
    """Interaktivní vizualizace vztahů mezi parametry."""
    print("\nVytváření pair plot pro vizualizaci vztahů mezi všemi parametry...")
    
    # Vytvoření pair plotu pro vizualizaci všech dvojic parametrů
    plt.figure(figsize=(16, 14))
    g = sns.pairplot(df, vars=numeric_columns, hue='rezim', palette='viridis', 
                    diag_kind='kde', plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'w'}, 
                    height=2.5)
    g.fig.suptitle('Vztahy mezi všemi parametry plazmatu podle režimu', fontsize=16, y=1.02)
    plt.show()

def visualize_stability_prediction(X_scaled, df, numeric_columns):
    """Vizualizace přesnosti předpovědi stability."""
    print("\nVizualizace přesnosti předpovědi stability...")
    
    # Připravíme data pro klasifikaci
    X_stability = X_scaled
    y_stability = df['stabilita']
    
    # Rozdělení na trénovací a testovací data
    X_train, X_test, y_train, y_test = train_test_split(X_stability, y_stability, test_size=0.3, random_state=42)
    
    # Trénování Random Forest klasifikátoru
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    # Predikce na testovacích datech
    y_pred = rf_classifier.predict(X_test)
    y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]  # Pravděpodobnost pozitivní třídy
    
    # Vytvoření vizualizačního dashboardu pro vyhodnocení modelu
    plt.figure(figsize=(18, 12))
    plt.suptitle('Vyhodnocení predikce stability plazmatu', fontsize=16)
    
    # 1. Confusion Matrix
    plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predikovaná hodnota')
    plt.ylabel('Skutečná hodnota')
    
    # 2. ROC křivka
    plt.subplot(2, 3, 2)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC křivka (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC křivka')
    plt.legend(loc="lower right")
    
    # 3. Precision-Recall křivka
    plt.subplot(2, 3, 3)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall křivka')
    
    # 4. Distribuce pravděpodobností
    plt.subplot(2, 3, 4)
    sns.histplot(y_pred_proba, bins=20, kde=True, color='green')
    plt.axvline(x=0.5, color='red', linestyle='--')
    plt.xlabel('Pravděpodobnost stability')
    plt.ylabel('Počet vzorků')
    plt.title('Distribuce pravděpodobností stability')
    
    # 5. Porovnání skutečných a predikovaných hodnot
    plt.subplot(2, 3, 5)
    plt.scatter(y_test, y_pred_proba, alpha=0.7)
    plt.xlabel('Skutečná hodnota')
    plt.ylabel('Predikovaná pravděpodobnost')
    plt.title('Skutečné vs. Predikované hodnoty')
    plt.grid(True)
    
    # 6. Feature importance
    plt.subplot(2, 3, 6)
    importances = rf_classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.bar(range(X_stability.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_stability.shape[1]), [numeric_columns[i] for i in indices], rotation=45)
    plt.title('Důležitost parametrů pro predikci stability')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    # Výpis klasifikačního reportu
    print("\nKlasifikační report předpovědi stability plazmatu:")
    print(classification_report(y_test, y_pred, target_names=['Nestabilní', 'Stabilní']))
    
    return y_test, y_pred, roc_auc, importances, indices

def summarize_findings(importances, indices, numeric_columns, cluster_labels, n_clusters, contingency_table, y_test, y_pred, roc_auc):
    """Shrnutí a závěrečná analýza."""
    print("\nShrnutí nejdůležitějších zjištění z vizualizací...")
    
    print("1. Nejdůležitější parametry pro stabilitu plazmatu:")
    for i, idx in enumerate(indices[:3]):
        print(f"   {i+1}. {numeric_columns[idx]} (důležitost: {importances[idx]:.3f})")
    
    print("\n2. Identifikované režimy plazmatu pomocí AI:")
    for i in range(n_clusters):
        cluster_size = np.sum(cluster_labels == i)
        main_regime = contingency_table[i].idxmax()
        main_regime_pct = contingency_table[i].max() / cluster_size * 100
        print(f"   Cluster {i}: {cluster_size} vzorků, převážně {main_regime} ({main_regime_pct:.1f}%)")
    
    print("\n3. Přesnost předpovědi stability plazmatu:")
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"   Celková přesnost: {accuracy:.2%}")
    print(f"   AUC ROC: {roc_auc:.2f}")
    
    print("\nZávěr: Analýza dat z tokamaku pomocí AI vizualizačních technik")
    print("odhalila klíčové parametry pro stabilitu plazmatu a pomohla")
    print("identifikovat různé režimy plazmatu. Tyto informace mohou být")
    print("využity pro optimalizaci provozu tokamaku a dosažení stabilního plazmatu.")

def main():
    """Hlavní funkce pro spuštění všech analýz."""
    # Načtení dat
    df = load_data('tokamak_data_cleaned.csv')
    
    # Příprava dat pro vizualizace
    numeric_columns = ['teplota', 'proud', 'magnet_pole', 'hustota', 'napeti', 'beta_param']
    X = df[numeric_columns]
    
    # Standardizace dat pro AI vizualizace
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Pokročilé časové vizualizace
    create_time_series_visualizations(df)
    
    # 2. Analýza korelací a vztahů mezi parametry
    importances = analyze_correlations(df, numeric_columns)
    
    # 3. Dimenzionální redukce pro vizualizaci vícerozměrných dat
    pca, pca_result = dimension_reduction_visualization(X_scaled, df)
    
    # 4. Identifikace režimů plazmatu pomocí shlukovací analýzy
    cluster_labels, n_clusters, contingency_table = clustering_analysis(X_scaled, pca_result, df)
    
    # 5. Trojrozměrná vizualizace fázového prostoru plazmatu
    create_3d_visualization(df)
    
    # 6. Analýza stabilních oblastí v parametrickém prostoru
    analyze_stability_regions(df, numeric_columns, X_scaled, scaler)
    
    # 7. Animovaná časová vizualizace trajektorie v prostoru parametrů
    create_animated_trajectory(df, X_scaled)
    
    # 8. Interaktivní vizualizace vztahů mezi parametry
    create_pair_plot(df, numeric_columns)
    
    # 9. Předpověď stability na základě parametrů
    y_test, y_pred, roc_auc, importances, indices = visualize_stability_prediction(X_scaled, df, numeric_columns)
    
    # 10. Shrnutí a závěrečná analýza
    summarize_findings(importances, indices, numeric_columns, cluster_labels, n_clusters, 
                     contingency_table, y_test, y_pred, roc_auc)
