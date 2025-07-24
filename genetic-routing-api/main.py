from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import random
import math
     
# --- API Metadata ---
app = FastAPI(
    title="Vehicle Routing Problem (VRP) Optimization API",
    description="Uses a Genetic Algorithm to solve the Vehicle Routing Problem.",
    version="1.1.0"
)
origins = [
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- Pydantic Models for API Response Structure ---

class RotaDetalhada(BaseModel):
    """Define a estrutura de uma única rota para a resposta da API."""
    id_veiculo: int
    tipo_veiculo: str
    trajeto: List[int]
    demanda_total: int
    distancia_km: float
    custo_rota: float

class SolucaoVRP(BaseModel):
    """Define a estrutura da resposta final da API."""
    melhor_custo_total: float
    geracao_encontrado: int
    detalhes_rotas: List[RotaDetalhada]
    
class MensagemErro(BaseModel):
    """Modelo para quando uma solução não é encontrada."""
    detalhe: str

# =============================================================================
# ETAPA 2: CONFIGURAÇÃO DO CENÁRIO E PARÂMETROS
# =============================================================================

# Define uma semente para o gerador de números aleatórios.
# Isso garante que os mesmos clientes sejam gerados a cada execução,
# tornando os resultados comparáveis. Remova se quiser aleatoriedade total.
random.seed(42)

VEICULOS = [
    {'id': 1, 'tipo': 'pequeno', 'capacidade': 80, 'custo_km': 1.2},
    {'id': 2, 'tipo': 'médio', 'capacidade': 110, 'custo_km': 1.5},
    {'id': 3, 'tipo': 'grande', 'capacidade': 150, 'custo_km': 2.0}
]
DEPOSITO = {'id': 0, 'x': 50, 'y': 50}

# Parâmetros do Algoritmo Genético
NUM_CLIENTES = 20
MAPA_X = 100
MAPA_Y = 100
TAMANHO_POPULACAO = 150
NUM_GERACOES = 1000  # Reduzido para uma resposta mais rápida na API
TAXA_MUTACAO = 0.5
NUM_ELITE = 15

# Geração de dados (clientes e mapa)
def gerar_clientes(num_clientes, mapa_x, mapa_y):
    clientes = []
    for i in range(1, num_clientes + 1):
        clientes.append({
            'id': i, 'x': random.randint(5, mapa_x - 5), 'y': random.randint(5, mapa_y - 5),
            'demanda': random.randint(5, 15)
        })
    return clientes

LISTA_CLIENTES = gerar_clientes(NUM_CLIENTES, MAPA_X, MAPA_Y)
MAPA_DADOS = {cliente['id']: cliente for cliente in LISTA_CLIENTES}
MAPA_DADOS[0] = DEPOSITO

# =============================================================================
# ETAPA 3: FUNÇÕES DO ALGORITMO GENÉTICO
# =============================================================================

def calcular_distancia(id1, id2):
    ponto1 = MAPA_DADOS[id1]
    ponto2 = MAPA_DADOS[id2]
    return math.sqrt((ponto1['x'] - ponto2['x'])**2 + (ponto1['y'] - ponto2['y'])**2)

def calcular_custo_rota(rota, veiculo):
    if not rota:
        return 0
    distancia_total = calcular_distancia(DEPOSITO['id'], rota[0])
    for i in range(len(rota) - 1):
        distancia_total += calcular_distancia(rota[i], rota[i+1])
    distancia_total += calcular_distancia(rota[-1], DEPOSITO['id'])
    return distancia_total * veiculo['custo_km']

def criar_individuo():
    ids_clientes = [cliente['id'] for cliente in LISTA_CLIENTES]
    random.shuffle(ids_clientes)
    return ids_clientes

def criar_populacao_inicial():
    return [criar_individuo() for _ in range(TAMANHO_POPULACAO)]

def calcular_fitness(individuo):
    veiculos_embaralhados = VEICULOS[:]
    random.shuffle(veiculos_embaralhados)
    rotas_por_veiculo = {}
    clientes_alocados = set()
    for veiculo in veiculos_embaralhados:
        rota_atual = []
        demanda_atual = 0
        rotas_por_veiculo[veiculo['id']] = rota_atual
        for id_cliente in individuo:
            if id_cliente not in clientes_alocados:
                demanda_cliente = MAPA_DADOS[id_cliente]['demanda']
                if demanda_atual + demanda_cliente <= veiculo['capacidade']:
                    rota_atual.append(id_cliente)
                    demanda_atual += demanda_cliente
                    clientes_alocados.add(id_cliente)

    if len(clientes_alocados) < NUM_CLIENTES:
        return 0, rotas_por_veiculo

    custo_total = sum(calcular_custo_rota(rota, next(v for v in VEICULOS if v['id'] == id_v)) for id_v, rota in rotas_por_veiculo.items())
    return 1 / (1 + custo_total), rotas_por_veiculo

def selecionar_por_torneio(populacao_avaliada):
    participantes = random.sample(populacao_avaliada, 5)
    participantes.sort(key=lambda item: item[1], reverse=True)
    return participantes[0][0]

def reproduzir(pai1, pai2):
    tamanho = len(pai1)
    ponto1, ponto2 = sorted(random.sample(range(tamanho), 2))
    filho = [None] * tamanho
    filho[ponto1:ponto2] = pai1[ponto1:ponto2]
    genes_pai2 = [gene for gene in pai2 if gene not in filho]
    idx_filho = 0
    for gene in genes_pai2:
        while filho[idx_filho] is not None:
            idx_filho += 1
        filho[idx_filho] = gene
    return filho

def aplicar_mutacao(individuo):
    if random.random() < TAXA_MUTACAO:
        i, j = random.sample(range(len(individuo)), 2)
        individuo[i], individuo[j] = individuo[j], individuo[i]
    return individuo

# =============================================================================
# ETAPA 4: FUNÇÃO PRINCIPAL DE RESOLUÇÃO
# =============================================================================

def resolver_vrp():
    populacao = criar_populacao_inicial()
    melhor_solucao_global = None
    menor_custo_global = float('inf')
    geracao_melhor_custo = -1

    for i in range(NUM_GERACOES):
        populacao_avaliada = []
        for individuo in populacao:
            fitness, rotas = calcular_fitness(individuo)
            custo = (1 / fitness) - 1 if fitness > 0 else float('inf')
            populacao_avaliada.append((individuo, fitness, custo, rotas))

        populacao_avaliada.sort(key=lambda item: item[1], reverse=True)
        
        custo_atual = populacao_avaliada[0][2]
        if custo_atual < menor_custo_global:
            menor_custo_global = custo_atual
            melhor_solucao_global = populacao_avaliada[0][3]
            geracao_melhor_custo = i

        nova_populacao = [ind[0] for ind in populacao_avaliada[:NUM_ELITE]]
        while len(nova_populacao) < TAMANHO_POPULACAO:
            pai1 = selecionar_por_torneio(populacao_avaliada)
            pai2 = selecionar_por_torneio(populacao_avaliada)
            filho = reproduzir(pai1, pai2)
            filho = aplicar_mutacao(filho)
            nova_populacao.append(filho)
        
        populacao = nova_populacao

    if melhor_solucao_global is None:
        return None

    return {
        "menor_custo_global": menor_custo_global,
        "geracao_melhor_custo": geracao_melhor_custo,
        "melhor_solucao_global": melhor_solucao_global
    }

# =============================================================================
# ETAPA 5: ROTAS DA API
# =============================================================================

@app.get("/", summary="Mensagem de Boas-vindas")
def root():
    return {"message": "Bem-vindo à API de Otimização de Rotas (VRP) com Algoritmo Genético"}

@app.get("/solve", 
         response_model=SolucaoVRP,
         summary="Executa o Algoritmo Genético para encontrar a melhor rota",
         responses={404: {"model": MensagemErro, "description": "Solução não encontrada"}})
def solve():
    """
    Executa o algoritmo genético para resolver o Problema de Roteamento de Veículos.
    
    Retorna a melhor solução encontrada, incluindo o custo total, a geração em que foi
    encontrada e os detalhes de cada rota.
    """
    resultado_execucao = resolver_vrp()

    if not resultado_execucao:
        return {"detalhe": "Não foi possível encontrar uma solução viável com os parâmetros atuais."}

    custo_total_final = 0
    rotas_detalhadas = []
    
    solucao = resultado_execucao["melhor_solucao_global"]
    
    for id_veiculo, rota in solucao.items():
        if not rota:
            continue
            
        veiculo_usado = next(v for v in VEICULOS if v['id'] == id_veiculo)
        
        demanda = sum(MAPA_DADOS[cid]['demanda'] for cid in rota)
        
        distancia = calcular_distancia(DEPOSITO['id'], rota[0])
        for i in range(len(rota) - 1):
            distancia += calcular_distancia(rota[i], rota[i+1])
        distancia += calcular_distancia(rota[-1], DEPOSITO['id'])
        
        custo = distancia * veiculo_usado['custo_km']
        custo_total_final += custo

        rotas_detalhadas.append(
            RotaDetalhada(
                id_veiculo=id_veiculo,
                tipo_veiculo=veiculo_usado['tipo'],
                trajeto=rota,
                demanda_total=demanda,
                distancia_km=round(distancia, 2),
                custo_rota=round(custo, 2)
            )
        )

    return SolucaoVRP(
        melhor_custo_total=round(custo_total_final, 2),
        geracao_encontrado=resultado_execucao["geracao_melhor_custo"],
        detalhes_rotas=rotas_detalhadas
    )

