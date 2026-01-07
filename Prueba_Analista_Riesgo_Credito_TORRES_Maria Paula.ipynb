#Importacion de datos 
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

# pip install xlrd para archivos excel antiguos 
dic21 = pd.read_excel("ig_2021_12.xls",header=None)
dic22 = pd.read_excel("ig_2022_12.xls",header=None)
dic23 = pd.read_excel("ig_2023_12.xls",header=None)
dic24 = pd.read_excel("ig_2024_12.xls",header=None)
sept24 = pd.read_excel("ig_2024_09.xls",header=None)
sept25 = pd.read_excel("ig_2025_09.xls",header=None)

#Generacion del codigo de limpieza de datos para 1 base
print(dic21.iloc[:15, :5]) #Identificacion de datos

dic21 = dic21.iloc[10:, :]
dic21.rename(columns={dic21.columns[0]: "rubro",
                   dic21.columns[1]: "subrubro",
                   dic21.columns[2]: "detalle"}, inplace=True)
# Rubro y subrubro para cada observación
dic21['rubro'] = dic21['rubro'].ffill()
dic21['subrubro'] = dic21['subrubro'].ffill()

#Creacion de la columna Fecha 
fechas = dic21.iloc[1, 3:]
fechas=fechas.dropna().unique()
print(fechas)

#Encabezados 
Encabezados=dic21.iloc[0, 3:].values
dic21.columns = (
    list(dic21.columns[:3]) +
    list(Encabezados)
)

dic21["fecha"] = fechas[0]
dic21 = dic21.iloc[2:, :].reset_index(drop=True)
dic21 = dic21[["fecha"] + [col for col in dic21.columns if col != "fecha"]]


#Automatización del código para todas las bases de datos 
def procesar_bases(base):

    base = base.iloc[10:, :].copy ()

    base.rename(columns={
        base.columns[0]: "rubro",
        base.columns[1]: "subrubro",
        base.columns[2]: "detalle"
    }, inplace=True)

    base["rubro"] = base["rubro"].ffill()
    base["subrubro"] = base["subrubro"].ffill()

    fechas = base.iloc[1, 3:]
    fechas = fechas.dropna().unique()

    
    Encabezados_G = base.iloc[0, 3:].values
    base.columns = (
        ["rubro", "subrubro", "detalle"] +
        list(Encabezados_G)
    )
    
    base["fecha"] = fechas[0]

    base = base.iloc[2:, :].reset_index(drop=True)
    base = base[["fecha", "rubro", "subrubro", "detalle"] +
                [c for c in base.columns if c not in ["fecha", "rubro", "subrubro", "detalle"]]]
    return base
    
archivos = [
    dic22,
    dic23,
    sept24,
    dic24,
    sept25,
]

bases = []

for base in archivos:
    base_def = procesar_bases(base)
    bases.append(base_def)

# Union bases de datos 
Indicadores = pd.concat(bases, ignore_index=True)
Indicadores=pd.concat([dic21,Indicadores],ignore_index=True)

# Unificación formato Fecha 

meses = ['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic']
Indicadores['fecha'] = Indicadores['fecha'].apply(
    lambda x: pd.Timestamp(year=2000+int(x.split('-')[1]), month=meses.index(x[:3].lower())+1, day=1)
    if isinstance(x, str) and '-' in x else x
)

Indicadores['fecha'] = pd.to_datetime(Indicadores['fecha'], errors='ignore').dt.date

#Selección columnas emisor elegido- BANCO MUNDO MUJER 

Indicadores = Indicadores[ 
    ['fecha', 'rubro', 'subrubro', 'detalle'] + 
    [col for col in Indicadores.columns if 'BANCO MUNDO MUJER S.A.' in str(col)]
]
#Indicadores.to_excel("indicadores_consolidados.xlsx", index=False)

# Selección de datos para graficar

# Rentabilidad
ROE=Indicadores[Indicadores['subrubro'] == 'UTILIDAD/PATRIMONIO']
ROA=Indicadores[Indicadores['subrubro'] == 'UTILIDAD/ACTIVO']

#Gráfico de barras doble eje 
fig, ax1 = plt.subplots(figsize=(10,6))
ax1.plot(ROE['fecha'], ROE['BANCO MUNDO MUJER S.A.'], marker='o', color='green', label='ROE')
ax1.set_xlabel('Fecha')
ax1.set_ylabel('ROE (%)', color='green')
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax1.tick_params(axis='y', labelcolor='green')

ax2 = ax1.twinx()
ax2.plot(ROA['fecha'], ROA['BANCO MUNDO MUJER S.A.'], marker='o', color='darkblue', label='ROA')
ax2.set_ylabel('ROA (%)', color='darkblue')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax2.tick_params(axis='y', labelcolor='darkblue')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
plt.title('ROE y ROA BANCO MUNDO MUJER S.A.')
ax1.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
#plt.show()

fig.savefig("ROE_ROA.png", dpi=300)


# CARTERA 
Indicadores['fecha'] = pd.to_datetime(Indicadores['fecha'], errors='coerce', dayfirst=True)

total_cartera= Indicadores[
    (Indicadores['rubro'] == 'INDICADORES DE RIESGO POR TIEMPO') &
    (Indicadores['subrubro'] == 'TOTAL CARTERA')
]
total_cartera = total_cartera.groupby('fecha', group_keys=False).head(7)

# Cartera vigente 
vigente_bruto = total_cartera.loc[
    (total_cartera['detalle'] == 'Vigente / Bruto') &
    ~((total_cartera['fecha'].dt.year == 2024) & (total_cartera['fecha'].dt.month == 9))
].copy()
vigente_bruto = vigente_bruto.dropna()
vigente_bruto = vigente_bruto.sort_values('fecha')

# Gráfico
plt.figure(figsize=(8,5))
plt.plot(vigente_bruto['fecha'].dt.strftime('%Y-%m'), vigente_bruto['BANCO MUNDO MUJER S.A.'],
    color='green', marker='o')
plt.title('Cartera Vigente BANCO MUNDO MUJER S.A.', fontsize=14)
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Vigente/Bruto', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)  # cuadrícula horizontal
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()
fig.savefig("Cartera Vigente.png", dpi=300)

# Cartera vencida
bruto = total_cartera.loc[
    (total_cartera['detalle'] == 'Bruto') &
    ~((total_cartera['fecha'].dt.year == 2024) & (total_cartera['fecha'].dt.month == 9))
].copy()

vencido = total_cartera.loc[
    (total_cartera['detalle'] == 'Vencido') &
    ~((total_cartera['fecha'].dt.year == 2024) & (total_cartera['fecha'].dt.month == 9))
].copy()

bruto = bruto.dropna(subset=['BANCO MUNDO MUJER S.A.'])
vencido = vencido.dropna(subset=['BANCO MUNDO MUJER S.A.'])
bruto = bruto.sort_values('fecha').reset_index(drop=True)
vencido = vencido.sort_values('fecha').reset_index(drop=True)
vencido_bruto= bruto[['fecha']].copy()
vencido_bruto['Vencido/Bruto'] = vencido['BANCO MUNDO MUJER S.A.'].values / bruto['BANCO MUNDO MUJER S.A.'].values

# Gráfico 
plt.plot(vencido_bruto['fecha'], vencido_bruto['Vencido/Bruto'], marker='o', linestyle='-', color='red')

for xi, yi in zip(vencido_bruto['fecha'], vencido_bruto['Vencido/Bruto']):
    plt.text(xi, yi, f"{yi*100:.2f}%", ha='center', va='bottom', fontsize=9)

plt.ylabel('Vencido / Bruto (%)')
plt.title('Cartera Vencida BANCO MUNDO MUJER')
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()
fig.savefig("Cartera Vencida.png", dpi=300)

#Cobertura 

cubrimiento = total_cartera.loc[
    (total_cartera['detalle'] == 'Cubrimiento') &
    ~((total_cartera['fecha'].dt.year == 2024) & (total_cartera['fecha'].dt.month == 9))
].copy()

cubrimiento= cubrimiento.dropna()
cubrimiento['fecha'] = pd.to_datetime(cubrimiento['fecha'])
cubrimiento= cubrimiento.sort_values('fecha')

#Gráfico
plt.figure(figsize=(10,5))

plt.plot(cubrimiento['fecha'],cubrimiento['BANCO MUNDO MUJER S.A.']*100, marker='o', linestyle='-', color='green')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

plt.title('Cubrimiento de la Cartera - BANCO MUNDO MUJER S.A.')
plt.xlabel('Fecha')
plt.ylabel('Cubrimiento (%)')
plt.xticks(cubrimiento['fecha'], cubrimiento['fecha'].dt.strftime('%Y-%m'))
plt.tight_layout()
for xi, yi in zip(cubrimiento['fecha'], cubrimiento['BANCO MUNDO MUJER S.A.']):
    plt.text(xi, yi*100, f"{yi*100:.1f}%", ha='center', va='bottom', fontsize=9)
#plt.show()

fig.savefig("Cobertura.png", dpi=300)


# Composicion 
subrubros_productivos = [
    "CARTERA COMERCIAL PRODUCTIVA",
    "CARTERA CONSUMO PRODUCTIVA",
    "CARTERA VIVIENDA PRODUCTIVA",
    "CARTERA MICROCREDITO PRODUCTIVA"
]

productivos = Indicadores[
    (Indicadores['rubro'] == 'INDICADORES DE RIESGO POR TIEMPO') &  (Indicadores['subrubro'].isin(subrubros_productivos + ["CARTERA Y LEASING PRODUCTIVO"]))
].copy()

productivos['fecha'] = pd.to_datetime(productivos['fecha'])

productivos = productivos[~((productivos['fecha'].dt.year == 2024) & (productivos['fecha'].dt.month == 9))]

tool = productivos.pivot_table(
    index='fecha',
    columns='subrubro',
    values='BANCO MUNDO MUJER S.A.'
)

tool[subrubros_productivos] = tool[subrubros_productivos].div(tool["CARTERA Y LEASING PRODUCTIVO"], axis=0)
tool = tool[subrubros_productivos]

# Cartera improductiva 
subrubros_improductivos = [
    "CARTERA COMERCIAL IMPRODUCTIVA",
    "CARTERA CONSUMO IMPRODUCTIVA",
    "CARTERA VIVIENDA IMPRODUCTIVA",
    "CARTERA MICROCREDITO IMPRODUCTIVA"
]

improductiva = Indicadores[
    (Indicadores['rubro'] == 'INDICADORES DE RIESGO POR TIEMPO') &
    (Indicadores['subrubro'].isin(subrubros_improductivos + ["CARTERA Y LEASING IMPRODUCTIVO"]))
].copy()
improductiva['fecha'] = pd.to_datetime(improductiva['fecha'])
improductiva = improductiva[~((improductiva['fecha'].dt.year == 2024) & (improductiva['fecha'].dt.month == 9))]
tool_improductiva = improductiva.pivot_table(index='fecha', columns='subrubro', values='BANCO MUNDO MUJER S.A.')
tool_improductiva[subrubros_improductivos] = tool_improductiva[subrubros_improductivos].div(tool_improductiva["CARTERA Y LEASING IMPRODUCTIVO"], axis=0)
tool_improductiva = tool_improductiva[subrubros_improductivos]

# GRAFICO DOBLE
leyenda_mapping = {
     "CARTERA COMERCIAL PRODUCTIVA": "Cartera Comercial",
    "CARTERA COMERCIAL IMPRODUCTIVA": "Cartera Comercial",
    "CARTERA CONSUMO PRODUCTIVA": "Cartera Consumo",
    "CARTERA CONSUMO IMPRODUCTIVA": "Cartera Consumo",
    "CARTERA VIVIENDA PRODUCTIVA": "Cartera Vivienda",
    "CARTERA VIVIENDA IMPRODUCTIVA": "Cartera Vivienda",
    "CARTERA MICROCREDITO PRODUCTIVA": "Cartera Microcrédito",
    "CARTERA MICROCREDITO IMPRODUCTIVA": "Cartera Microcrédito"
}
colores = {
    "Cartera Comercial": "darkgreen",
    "Cartera Consumo": "springgreen",
    "Cartera Vivienda": "seagreen",
    "Cartera Microcrédito": "limegreen"
}
# Graficar 2 paneles
tool = tool.astype(float).fillna(0).reset_index()
tool_improductiva = tool_improductiva.astype(float).fillna(0).reset_index()

fig, axs = plt.subplots(ncols=2, figsize=(16,6), sharey=True)

def plot_tool(ax, df, mapping, colores):
    bottom_vals = pd.Series(0, index=df.index)
    
    x_pos = range(len(df))  # posiciones numéricas para las barras
    x_labels = [d.strftime('%b %Y') for d in df['fecha']]      
    for col_original, nombre_leyenda in mapping.items():
        if col_original in df.columns:
            valores = df[col_original].astype(float)
            ax.bar(
                x_pos,
                valores,
                bottom=bottom_vals,
                color=colores[nombre_leyenda],
                edgecolor='black', 
                width=0.6
            )
            for i, (v, b) in enumerate(zip(valores, bottom_vals)):
                if v > 0:
                    ax.text(
                        i, 
                        b + v / 2,   
                        f"{v*100:.1f}%", 
                        ha='center', 
                        va='center', 
                        fontsize=9,
                        color='white'  
                    )
            bottom_vals += valores
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))

#Improductiva
plot_tool(axs[0], tool_improductiva, leyenda_mapping, colores)
axs[0].set_title("Cartera Improductiva")
axs[0].set_xlabel("Fecha")
axs[0].set_ylabel("Porcentaje del Total (%)")

#Productiva
plot_tool(axs[1], tool, leyenda_mapping, colores)
axs[1].set_title("Cartera Productiva")
axs[1].set_xlabel("Fecha")

handles = [plt.Rectangle((0,0),1,1, color=colores[n]) for n in colores]
labels = list(colores.keys())
axs[1].legend(handles, labels, title="Composición", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
#plt.show()
fig.savefig("COMPOSICION.png", dpi=300)

# Productiva vs improductiva 

productivo_total = productivos[productivos['subrubro'] == "CARTERA Y LEASING PRODUCTIVO"][['fecha','BANCO MUNDO MUJER S.A.']].copy()
improductivo_total = improductiva[improductiva['subrubro'] == "CARTERA Y LEASING IMPRODUCTIVO"][['fecha','BANCO MUNDO MUJER S.A.']].copy()
productivo_total = productivo_total.rename(columns={'BANCO MUNDO MUJER S.A.':'Productivo'})
improductivo_total = improductivo_total.rename(columns={'BANCO MUNDO MUJER S.A.':'Improductivo'})

fechas = sorted(set(productivo_total['fecha']).union(set(improductivo_total['fecha'])))
#Base gráfico 
Total = pd.DataFrame({'fecha': fechas})
Total = Total.merge(productivo_total, on='fecha', how='left')
Total = Total.merge(improductivo_total, on='fecha', how='left')
Total = Total.fillna(0)

Total['Total'] = Total['Productivo'] + Total['Improductivo']
Total['Productivo %'] = Total['Productivo'] / Total['Total']
Total['Improductivo %'] = Total['Improductivo'] / Total['Total']
Total['fecha'] = pd.to_datetime(Total['fecha'])


#Gráfico de barras comparativo 

fig, ax = plt.subplots(figsize=(12,6))
width = pd.Timedelta(days=60)  

ax.bar(Total['fecha'] - width/2, Total['Productivo %'], width=width, color='green', edgecolor='black', label='Productivo')
ax.bar(Total['fecha'] + width/2, Total['Improductivo %'], width=width, color='limegreen', edgecolor='black', label='Improductivo')
for i, row in Total.iterrows():
    if row['Productivo %'] > 0:
        ax.text(row['fecha'] - width/2, row['Productivo %']/2, f"{row['Productivo %']*100:.1f}%", 
                ha='center', va='center', color='white', fontsize=10)
    if row['Improductivo %'] > 0:
        ax.text(row['fecha'] + width/2, row['Improductivo %']/2, f"{row['Improductivo %']*100:.1f}%", 
                ha='center', va='center', color='white', fontsize=10)


ax.set_xticks(Total['fecha'])
ax.set_xticklabels([d.strftime('%b %Y') for d in Total['fecha']], rotation=45, ha='right')
ax.set_ylabel("Porcentaje (%)")
ax.set_title("Cartera Productiva vs Improductiva (%)")
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
ax.legend()

fig.autofmt_xdate()  
plt.tight_layout()
#plt.show()

fig.savefig("Productiva e impro.png", dpi=300)

# Apalancamiento

apalancamiento=Indicadores[Indicadores['subrubro'] == 'EXPOSICION PATRIMONIAL (SIN PROPIEDADES Y EQUIPO)']

plt.plot(apalancamiento['fecha'], apalancamiento['BANCO MUNDO MUJER S.A.'], marker='o', color="green")

for x, y in zip(apalancamiento['fecha'], apalancamiento['BANCO MUNDO MUJER S.A.']):
    plt.text(x, y + 0.01, f"{y*100:.1f}%", ha='center', va='bottom', fontsize=9, color='green')

plt.title('Exposición Patrimonial BANCO MUNDO MUJER S.A.')
plt.xlabel('Fecha')
plt.ylabel('Exposición Patrimonial')
plt.ylim(0.22, 0.5)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.grid(True, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)  
plt.tight_layout()
#plt.show()
plt.savefig("Exposición Patrimonia.png", dpi=300) 
