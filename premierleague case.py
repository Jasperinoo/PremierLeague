script
# %%
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

# %%
df = pd.read_csv("PremierLeague.csv")
df

# %%
players = pd.read_csv("players.csv")
transfers = pd.read_csv("transfers.csv")
game_lineups = pd.read_csv("game_lineups.csv")
games = pd.read_csv("games.csv")
player_valuations = pd.read_csv("player_valuations.csv")
game_events = pd.read_csv("game_events.csv")
club_games = pd.read_csv("club_games.csv")
clubs = pd.read_csv("clubs.csv")
competitions = pd.read_csv("competitions.csv")
appearances = pd.read_csv("appearances.csv")

print(players.head())

# %%
import requests
API_KEY = 	"703e38c5af704ea2b71e33878e34d5c4"
url = 'https://api.football-data.org/v4/competitions/PL/teams'
headers = {
   'X-Auth-Token': API_KEY
}
response = requests.get(url, headers=headers)
if response.status_code == 200:
   data = response.json()
   for team in data['teams']:
       print(team['name'])
else:
   print(f'Fout: {response.status_code}, {response.text}')

# %% [markdown]
# Code om NaN-waarden te verwijderen:

# %%
players = players.dropna()
transfers = transfers.dropna()
game_lineups = game_lineups.dropna()
games = games.dropna()
player_valuations = player_valuations.dropna()
game_events = game_events.dropna()
club_games = club_games.dropna()
clubs = clubs.dropna()
competitions = competitions.dropna()
appearances = appearances.dropna()

# %% [markdown]
# Code om NaN-waarden te verwijderen:

# %%


API_KEY = "703e38c5af704ea2b71e33878e34d5c4"
url = 'https://api.football-data.org/v4/competitions/PL/teams'
headers = {'X-Auth-Token': API_KEY}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    

    teams_df = pd.DataFrame(data['teams'])
    

    teams_df_clean = teams_df.dropna()
    
  
    print(teams_df_clean.head())
else:
    print(f'Fout: {response.status_code}, {response.text}')


# %% [markdown]
# Code om NaN-waarden te verwijderen:

# %%

df = pd.read_csv("PremierLeague.csv")

df_clean = df.dropna()

print(df_clean.head())


# %%
print(games.columns)

# %% [markdown]
# 
# 
# Variabele full time uitslag 

# %%
games['FullTimeResult'] = games.apply(
    lambda row: 'H' if row['home_club_goals'] > row['away_club_goals']
    else 'A' if row['home_club_goals'] < row['away_club_goals']
    else 'D', 
    axis=1
)

print(games[['home_club_goals', 'away_club_goals', 'FullTimeResult']].head())

# %% [markdown]
# Koppeling op clubniveau: 
# 
# PremierLeague.csv → bevat HomeTeam en AwayTeam
# 
# Transfermarkt (clubs.csv) → bevat club_id en club_name
# 
# Football-Data API → bevat team_id en team_name

# %%
# Bekijk de kolomnamen in de clubs DataFrame
print(clubs.columns)


# %%
# Merge Premier League data met clubs.csv om club_id toe te voegen
games = games.merge(clubs[['club_id', 'name']], left_on='home_club_name', right_on='name', how='left')
games = games.merge(clubs[['club_id', 'name']], left_on='away_club_name', right_on='name', how='left', suffixes=('_home', '_away'))

# Bekijk het resultaat
print(games.head())


# %%
# Stel dat het competition_id voor de Premier League bijvoorbeeld 1 is
premier_league_id = 1  # Pas dit aan naar het juiste ID van de Premier League

# Filter de games op Premier League
games = games[games['competition_id'] == premier_league_id]

# Bekijk het resultaat
print(games.head())


# %%
# Bekijk de kolomnamen van de transfers dataframe
print(transfers.columns)


# %% [markdown]
# Stap 1: Bereken inkomende en uitgaande transfers

# %%
# Tel inkomende en uitgaande transfers per club
incoming_transfers = transfers.groupby('to_club_id').size().reset_index(name='incoming_transfers')
outgoing_transfers = transfers.groupby('from_club_id').size().reset_index(name='outgoing_transfers')

# Bekijk de eerste paar rijen om te controleren
print(incoming_transfers.head())
print(outgoing_transfers.head())


# %% [markdown]
# Merge de gegevens met de games dataset

# %%
# Merge met Premier League teams
games = games.merge(incoming_transfers, left_on='home_club_id', right_on='to_club_id', how='left')
games = games.merge(outgoing_transfers, left_on='home_club_id', right_on='from_club_id', how='left')

# Bekijk het resultaat
print(games.head())


# %%
print(games.columns)


# %% [markdown]
# Koppeling met API-data (huidige prestaties en teams):

# %%


# API-key en URL
API_KEY = "3ce4d66c346340f09c3a16c445987ca4"
url = 'https://api.football-data.org/v4/competitions/PL/teams'
headers = {'X-Auth-Token': API_KEY}

# Verkrijg de gegevens van de API
response = requests.get(url, headers=headers)
if response.status_code == 200:
    api_data = response.json()
    api_teams = pd.DataFrame(api_data['teams'])[['id', 'name']]  # Haal alleen de benodigde kolommen op

    # Hernoem de 'name' kolom naar 'api_team_name' om duplicaten te vermijden
    api_teams.rename(columns={'name': 'api_team_name'}, inplace=True)

    # Merge met games: gebruik name_home voor de naam van de thuisclub
    games = games.merge(api_teams, left_on='name_home', right_on='api_team_name', how='left')

# Bekijk het resultaat
print(games.head())



# %%
print(competitions.columns)


# %%
print(competitions.head())  # Bekijk de eerste paar rijen
print(competitions.sample(5))  # Bekijk 5 willekeurige rijen


# %%
premier_league = competitions[competitions["domestic_league_code"] == "GB1"]
print(premier_league)

# %% [markdown]
#  Laten we eerst controleren welke kolommen elke dataset heeft voordat we filteren. Dit geeft een overzicht van alle kolommen in je datasets. Zoek naar de juiste kolom die aangeeft tot welke competitie de data behoort .

# %%
datasets = {
    "games": games,
    "club_games": club_games,
    "game_events": game_events,
    "game_lineups": game_lineups,
    "appearances": appearances
}

# Check welke kolommen elke dataset heeft
for name, df in datasets.items():
    print(f"{name} columns: {df.columns.tolist()}\n")
    

# %% [markdown]
# Filter alleen Premier League-data uit Transfermarkt CSV's

# %%
# Alleen wedstrijden uit de Premier League selecteren
premier_league_games = games[games["competition_id"] == "GB1"]
premier_league_appearances = appearances[appearances["competition_id"] == "GB1"]

# Check of het goed gefilterd is
print(premier_league_games["competition_id"].unique())  # Moet alleen 'GB1' laten zien
print(premier_league_appearances["competition_id"].unique())  # Moet alleen 'GB1' laten zien



# %%
print(games["competition_id"].unique())


# %%
print(games.columns)


# %%
print(games["competition_id"].isna().sum())


# %%
print(games["competition_id"].dtype)


# %%
print(games["competition_id"].unique())


# %% [markdown]
# Stap 1: Verkrijg de Premier League gegevens uit competitions

# %%
import pandas as pd

# Laad de competities, clubs en transfers datasets
path = "C:/Users/dshab/Downloads/Transfermarkt/"
competitions = pd.read_csv(path + "competitions.csv")
clubs = pd.read_csv(path + "clubs.csv")
transfers = pd.read_csv(path + "transfers.csv")

# Filter de Premier League competitie op basis van de 'domestic_league_code' (GB1)
premier_league = competitions[competitions["domestic_league_code"] == "GB1"]

# Bekijk het 'competition_id' voor de Premier League
premier_league_id = premier_league['competition_id'].values[0]
print(premier_league)


# %%
print(clubs.columns)

# %%
# Bekijk de unieke waarden in de 'domestic_competition_id' kolom
print(clubs['domestic_competition_id'].unique())


# %% [markdown]
# Stap 2: Filter de clubs dataset voor Premier League-teams

# %%
# Filter clubs die behoren tot de Premier League (gebruik 'domestic_competition_id' uit 'clubs' en filter op 'GB1')
premier_league_clubs = clubs[clubs['domestic_competition_id'] == 'GB1']

# Bekijk de Premier League clubs
print(premier_league_clubs)




# %%
# Bekijk de kolommen in de games en competitions datasets
print(games.columns)
print(competitions.columns)



# %% [markdown]
# Stap 3: Filter de transfers dataset op basis van Premier League-clubs

# %%
# Filter de transfers dataset op basis van de Premier League-clubs
premier_league_transfers = transfers[
    (transfers['from_club_id'].isin(premier_league_clubs['club_id'])) |
    (transfers['to_club_id'].isin(premier_league_clubs['club_id']))
]

# Bekijk de gefilterde transfers
print(premier_league_transfers.head())


# %% [markdown]
# Code om NaN-waarden te verwijderen:

# %%
# Verwijder rijen met NaN-waarden in de belangrijke kolommen (zoals 'transfer_fee' en 'market_value_in_eur')
premier_league_transfers_clean = premier_league_transfers.dropna(subset=['transfer_fee', 'market_value_in_eur'])

# Bekijk de schone dataset
print(premier_league_transfers_clean.head())


# %% [markdown]
# 1. premier_league_transfers voor alle transfers die relevant zijn voor Premier League-clubs.
# 
# 2. De API voor clubnamen (om een lijst van de Premier League-teams te krijgen).
# 
# 3. PremierLeague.csv voor de wedstrijdresultaten van de Premier League.

# %% [markdown]
# Stap 1: Koppel de premier_league_transfers aan de clubnamen
# Laten we beginnen door de clubnamen toe te voegen aan de premier_league_transfers. We kunnen dat doen door de club-IDs te matchen met de juiste clubnamen. Aangezien we deze clubnamen eerder al van de API hebben gehaald, moeten we dat combineren met de premier_league_transfers.

# %%
# Koppelen van de clubnamen uit de API met de transferdata
# Eerst voegen we de clubnamen toe voor de "from_club_id" en "to_club_id"

premier_league_transfers = premier_league_transfers.merge(premier_league_clubs[['club_id', 'name']], 
                                                          left_on='from_club_id', right_on='club_id', 
                                                          how='left', suffixes=('_from', '_from_name'))

premier_league_transfers = premier_league_transfers.merge(premier_league_clubs[['club_id', 'name']], 
                                                          left_on='to_club_id', right_on='club_id', 
                                                          how='left', suffixes=('_to', '_to_name'))

# Nu hebben we de clubnamen gekoppeld aan de transferdata
print(premier_league_transfers[['player_name', 'from_club_name', 'to_club_name', 'transfer_fee', 'transfer_date']].head())


# %% [markdown]
# Stap 2: Merge de PremierLeague.csv (wedstrijden) met de premier_league_transfers
# 
# We kunnen nu proberen de PremierLeague.csv dataset te combineren met de premier_league_transfers om te zien of de transfers invloed hebben op de prestaties van teams in wedstrijden.
# 
# De belangrijkste kolommen voor de merge zijn waarschijnlijk de clubnamen en de seizoenen.

# %%
# Print de eerste paar rijen van de clubnamen in beide datasets om te controleren
print("HomeTeam uit Premier League:")
print(df_premier_league['HomeTeam'].head())  # Controleer de clubnamen in de wedstrijden dataset

print("to_club_name uit Transfers:")
print(premier_league_transfers['to_club_name'].head())  # Controleer de clubnamen in de transfers dataset




# %%
# Print de eerste paar seizoenen in beide datasets
print("Seizoen uit Premier League:")
print(df_premier_league['Season'].head())  # Controleer seizoenformaat in de Premier League dataset

print("Seizoen uit Transfers:")
print(premier_league_transfers['transfer_season'].head())  # Controleer seizoenformaat in de transfers dataset


# %%
# Zorg ervoor dat de seizoen format consistent is
premier_league_transfers['transfer_season'] = premier_league_transfers['transfer_season'].str[:4]  # Alleen het jaartal
df_premier_league['Season'] = df_premier_league['Season'].str[:4]  # Alleen het jaartal

# Zorg ervoor dat de clubnamen schoon zijn (zonder extra spaties)
df_premier_league['HomeTeam'] = df_premier_league['HomeTeam'].str.strip().str.lower()  # Verwijder spaties en zet om naar kleine letters
premier_league_transfers['to_club_name'] = premier_league_transfers['to_club_name'].str.strip().str.lower()  # Verwijder spaties en zet om naar kleine letters

# Controleer de eerste paar waarden van beide datasets na het schoonmaken
print("HomeTeam uit Premier League:")
print(df_premier_league['HomeTeam'].head())

print("to_club_name uit Transfers:")
print(premier_league_transfers['to_club_name'].head())

# Merge de wedstrijden met de transfers op seizoen en clubnaam
merged_data = pd.merge(df_premier_league, premier_league_transfers, 
                       left_on=['HomeTeam', 'Season'], 
                       right_on=['to_club_name', 'transfer_season'], 
                       how='left')

# Verwijder rijen waar belangrijke kolommen NaN zijn (zoals 'player_name', 'from_club_name', of 'to_club_name')
merged_data_cleaned = merged_data.dropna(subset=['player_name', 'from_club_name', 'to_club_name'])

# Bekijk de schoongewassen data
print(merged_data_cleaned.head())


# %%
# Controleer of de bestanden goed worden ingeladen
df_premier_league = pd.read_csv("PremierLeague/PremierLeague.csv")
print("Eerste paar rijen van de Premier League wedstrijden:")
print(df_premier_league.head())

# Controleer of de juiste kolomnamen aanwezig zijn in df_premier_league
print("Kolomnamen in Premier League wedstrijden:")
print(df_premier_league.columns)

# Bekijk de eerste paar rijen van de transfers data
print("Eerste paar rijen van de Premier League transfers:")
print(premier_league_transfers.head())

# Controleer de kolomnamen van de transfers data
print("Kolomnamen in Premier League transfers:")
print(premier_league_transfers.columns)

# Controleren op lege waarden voor seizoen en clubnamen in beide datasets
print("Aantal lege waarden in de 'Season' kolom van Premier League wedstrijden:")
print(df_premier_league['Season'].isna().sum())

print("Aantal lege waarden in de 'HomeTeam' kolom van Premier League wedstrijden:")
print(df_premier_league['HomeTeam'].isna().sum())

print("Aantal lege waarden in de 'transfer_season' kolom van Premier League transfers:")
print(premier_league_transfers['transfer_season'].isna().sum())

print("Aantal lege waarden in de 'to_club_name' kolom van Premier League transfers:")
print(premier_league_transfers['to_club_name'].isna().sum())




# %%
# Omzetten van de 'transfer_season' kolom naar het juiste formaat (bijvoorbeeld '2025-2026')
def format_season(season):
    year = int(season.split('/')[0]) + 2000  # Zorg ervoor dat we een jaar krijgen, zoals 2025 voor '25/2'
    return f"{year}-{year + 1}"

# Pas de functie toe op de transfer_season kolom
premier_league_transfers['transfer_season'] = premier_league_transfers['transfer_season'].apply(format_season)

# Controleer de nieuwe 'transfer_season' waarden
print("Nieuwe 'transfer_season' waarden:")
print(premier_league_transfers['transfer_season'].unique())

# Merge de datasets opnieuw, dit keer met de omgezette seizoenen
merged_data = pd.merge(df_premier_league, premier_league_transfers, left_on=['HomeTeam', 'Season'], 
                       right_on=['to_club_name', 'transfer_season'], how='left')

# Bekijk het resultaat van de merge
print(merged_data.head())


# %% [markdown]
# Wat deze code doet:
# Filteren van de seizoenen: We filteren de wedstrijden van de Premier League tussen 2020 en 2025. Hierdoor krijgen we meer relevante wedstrijden voor de transferperiode.
# Merge uitvoeren: We proberen nu de gefilterde dataset te combineren met de transferdata van de Premier League.

# %%
# Stap 1: Filter de Premier League wedstrijden dataset op seizoenen van 2020-2025
df_premier_league_filtered = df_premier_league[df_premier_league['Season'].str[:4].isin(['2020', '2021', '2022', '2023', '2024', '2025'])]

# Controleer de gefilterde data
print("Gefilterde wedstrijden (2020-2025):")
print(df_premier_league_filtered.head())

# Stap 2: Merge de gefilterde Premier League wedstrijden dataset met de transfer dataset
merged_data_filtered = pd.merge(df_premier_league_filtered, premier_league_transfers, 
                                left_on=['HomeTeam', 'Season'], right_on=['to_club_name', 'transfer_season'], how='left')

# Bekijk het resultaat van de merge
print("Merged Data (2020-2025):")
print(merged_data_filtered.head())


# %% [markdown]
# Wat de code doet:
# Formaat van het seizoen aanpassen: We zorgen ervoor dat we alleen het jaartal gebruiken (zoals 2020-2021 om te corresponderen met 2020).
# Merge: De gefilterde Premier League wedstrijden worden nu samengevoegd met de transfers op basis van zowel HomeTeam en Season uit de Premier League als to_club_name en transfer_season uit de transfers.

# %%
# Zorg ervoor dat 'Season' uit beide datasets het juiste formaat heeft (bijv. alleen het jaar zonder maand/ datum)
df_premier_league_filtered['Season'] = df_premier_league_filtered['Season'].str[:4]  # We gebruiken alleen het jaar

# Dezelfde bewerking voor 'transfer_season' kolom in premier_league_transfers
premier_league_transfers['transfer_season'] = premier_league_transfers['transfer_season'].str[:4]

# Merge de gefilterde Premier League wedstrijden dataset met de transfer dataset
merged_data_filtered = pd.merge(df_premier_league_filtered, premier_league_transfers, 
                                left_on=['HomeTeam', 'Season'], right_on=['to_club_name', 'transfer_season'], how='left')

# Bekijk de resultaten van de merge
print("Merged Data (2020-2025):")
print(merged_data_filtered.head())


# %% [markdown]
# "NaN"-waarden vervangen door betekenisvolle standaardwaarden.

# %%
# Vul NaN-waarden in de 'to_club_name' en 'transfer_fee' kolommen
merged_data_filtered_filled = merged_data_filtered.fillna({'to_club_name': 'Geen transfer', 'transfer_fee': 0})

# Bekijk de ingevulde data
print("Ingevulde Merged Data (2020-2025):")
print(merged_data_filtered_filled.head())


# %% [markdown]
# 1. Premier League Transfers Data voor transferinformatie.
# 
# 2. API voor Clubnamen voor het verkrijgen van actieve Premier League-teams.
# 
# 3. Premier League Wedstrijden Data voor het koppelen van teams aan hun prestaties in wedstrijden.

# %% [markdown]
# De titel is dynamisch en wordt bijgewerkt wanneer de slider wordt aangepast. Het laat duidelijk zien dat de grafiek het aantal overwinningen toont voor teams met een bepaald aantal inkomende transfers. 
# 
# De slider staat de gebruiker toe om het minimum aantal inkomende transfers te kiezen. De grafiek wordt automatisch bijgewerkt om alleen de teams te tonen die dit aantal of meer inkomende transfers hebben.De slider in de visualisatie filtert de teams op basis van het aantal inkomende transfers en toont vervolgens het aantal overwinningen van die teams. De grafiek verandert dynamisch op basis van de keuze van de gebruiker in de slider (bijvoorbeeld, als je het minimum aantal inkomende transfers instelt op 6, wordt alleen het aantal overwinningen voor teams die 6 of meer inkomende transfers hebben weergegeven).
# 
# De Y-as toont het aantal overwinningen van de geselecteerde teams.
# De X-as toont de teamnamen die voldoen aan de geselecteerde filter.
# 
# Hoe deze datasets gecombineerd werden:
# 
# Filtreren van transfers: In de Premier League Transfers Data werden transfers van spelers gefilterd op basis van seizoen en club, waarbij we gekeken hebben naar de transfers die relevant zijn voor de Premier League-teams.
# 
# Seizoensspecifieke filtering: De gegevens werden gefilterd op basis van het seizoen en dan werd de aantal inkomende transfers per team geteld. Dit gaf een indicatie van hoeveel transfers elk team had in een bepaald seizoen.
# 
# Wedstrijddata: De Premier League Wedstrijden dataset werd gebruikt om de teamresultaten te koppelen aan de teams in de transfers dataset. We konden dan het aantal overwinningen per team berekenen en vergelijken met het aantal inkomende transfers.
# 

# %% [markdown]
# Wat doet de code nu:
# De slider stelt je in staat om het minimum aantal inkomende transfers in te stellen.
# De grafiek toont vervolgens alleen de teams die dit minimum aantal inkomende transfers of meer hebben.
# De grafiek toont de aantal overwinningen van deze gefilterde teams.

# %% [markdown]
# SLIDER

# %%
import plotly.express as px
import pandas as pd
from ipywidgets import interact

# Voorbeelddata: Vervang deze met je eigen dataframe van Premier League teams
df = pd.DataFrame({
    'Team': ['Arsenal', 'Chelsea', 'Liverpool', 'Tottenham', 'Man City', 'Manchester United', 'Leicester City', 'Everton', 'West Ham', 'Aston Villa'],
    'Inkomende_Transfers': [5, 7, 3, 8, 6, 4, 2, 3, 4, 5],
    'Aantal_Overwinningen': [20, 18, 22, 15, 23, 19, 14, 12, 16, 13]
})

# Functie die de grafiek bijwerkt op basis van het aantal inkomende transfers
def update_graph(min_transfers):
    # Filter de data op basis van het minimum aantal inkomende transfers
    filtered_df = df[df['Inkomende_Transfers'] >= min_transfers]
    
    # Maak een grafiek van het aantal overwinningen
    fig = px.bar(filtered_df, x='Team', y='Aantal_Overwinningen',
                 title=f"Aantal Overwinningen van Teams met {min_transfers} of Meer Inkomende Transfers",
                 labels={'Aantal_Overwinningen': 'Aantal Overwinningen', 'Team': 'Team'},
                 color='Team',  # Zet de kleur op basis van het team, zodat de balken verschillend zijn
                 template="plotly",  # Gebruik een lichtere template
                 color_discrete_sequence=px.colors.qualitative.Set2)  # Kleurenschema voor verschillende teams
    
    # Zorg ervoor dat de y-as zich aanpast aan de hoogste waarde
    fig.update_layout(
        yaxis=dict(range=[0, filtered_df['Aantal_Overwinningen'].max() + 2]),  # Stel het bereik van de y-as in
        plot_bgcolor='white',  # Zet de achtergrondkleur naar wit
        title_font=dict(size=20, color='black'),  # Verander de kleur en grootte van de titel
        xaxis_title_font=dict(size=15, color='black'),  # Verander de kleur van de x-as titel
        yaxis_title_font=dict(size=15, color='black')   # Verander de kleur van de y-as titel
    )
    
    # Toon de grafiek
    fig.show()

# Maak de interactieve slider
interact(update_graph, min_transfers=(df['Inkomende_Transfers'].min(), df['Inkomende_Transfers'].max(), 1))




# %% [markdown]
# Volgende stap: 
# 
# Regressie-analyse toevoegen (voor checkbox)
# Nu dat we de basisvisualisatie hebben, kunnen we verder gaan met het toevoegen van een checkbox waarmee we een regressielijn kunnen tonen op basis van de inkomende transfers en het aantal overwinningen. De checkbox zal de gebruiker de mogelijkheid geven om te bepalen of ze de regressielijn willen zien, wat kan helpen om patronen te visualiseren en de relatie tussen inkomende transfers en overwinningen te onderzoeken.
# 
# De regressielijn kan laten zien of er een lineaire relatie is tussen deze twee variabelen. Regressielijn: De regressielijn laat de algemene trend zien tussen inkomende transfers en het aantal overwinningen voor de geselecteerde teams.

# %% [markdown]
# Wat kunnen we nu analyseren?
# 
# Met de regressielijn kunnen we zien of er een positieve of negatieve correlatie is tussen het aantal inkomende transfers en de prestaties van teams (overwinningen).
# Dit geeft ons een visueel inzicht in hoe de veranderingen in teamopstellingen door transfers mogelijk invloed hebben op hun prestaties.
# 
# 
# De datapunten laten de werkelijke prestaties van de teams zien, en de regressielijn probeert het algemene patroon te modelleren tussen het aantal inkomende transfers en het aantal overwinningen. Hoe dichter de datapunten bij de lijn liggen, hoe sterker de relatie is tussen de twee variabelen. Als de datapunten een grote spreiding vertonen, is de relatie minder duidelijk.

# %% [markdown]
# CHECKBOX (scatterplot)

# %%
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from ipywidgets import widgets, interact

# Voorbeelddata
df = pd.DataFrame({
    'Team': ['Arsenal', 'Chelsea', 'Liverpool', 'Tottenham', 'Man City', 'Manchester United', 'Leicester City', 'Everton', 'West Ham', 'Aston Villa'],
    'Inkomende_Transfers': [5, 7, 3, 8, 6, 4, 2, 3, 4, 5],
    'Aantal_Overwinningen': [20, 18, 22, 15, 23, 19, 14, 12, 16, 13]
})

# Functie die de scatterplot met optionele regressielijn en filtering bijwerkt
def update_graph(show_regression, enable_slider, min_transfers=2):
    # Filter de data als de slider actief is
    filtered_df = df[df['Inkomende_Transfers'] >= min_transfers] if enable_slider else df

    fig = px.scatter(
        filtered_df, x='Inkomende_Transfers', y='Aantal_Overwinningen',
        text='Team',
        hover_data={'Team': True, 'Inkomende_Transfers': True, 'Aantal_Overwinningen': True},
        title="Relatie tussen Inkomende Transfers en Aantal Overwinningen",
        labels={'Inkomende_Transfers': 'Aantal Inkomende Transfers', 'Aantal_Overwinningen': 'Aantal Overwinningen'},
        template="plotly",
        color='Team',
        color_discrete_sequence=px.colors.qualitative.Set2  # Consistente kleuren
    )

    fig.update_traces(
        marker=dict(size=10, opacity=0.8),
        textposition="top center"
    )

    # Voeg een regressielijn toe als de checkbox is aangevinkt
    if show_regression:
        trendline = px.scatter(filtered_df, x='Inkomende_Transfers', y='Aantal_Overwinningen', trendline="ols")
        regression_line = trendline.data[1]
        regression_line.line.color = 'black'
        regression_line.line.dash = 'dot'
        fig.add_trace(regression_line)

    fig.update_layout(
        yaxis=dict(range=[0, 25]),  # Y-as tot 25
        plot_bgcolor='white',
        title_font=dict(size=18, color='black'),
        xaxis_title_font=dict(size=14, color='black'),
        yaxis_title_font=dict(size=14, color='black')
    )

    fig.show()

# Interactieve checkboxes en slider
def interactive_plot(show_regression, enable_slider):
    if enable_slider:
        interact(update_graph, show_regression=widgets.fixed(show_regression), enable_slider=widgets.fixed(True),
                 min_transfers=(df['Inkomende_Transfers'].min(), df['Inkomende_Transfers'].max(), 1))
    else:
        update_graph(show_regression, False)

interact(interactive_plot, 
         show_regression=widgets.Checkbox(value=False, description="Toon Regressielijn"), 
         enable_slider=widgets.Checkbox(value=False, description="Activeer Slider voor Filtering"))



# %% [markdown]
# Verhouding van Inkomende Transfers tot Overwinningen

# %% [markdown]
# Verhouding= 
# Inkomende Transfers / Aantal Overwinningen
# ​
#  
# 

# %% [markdown]
# Wat gebeurt er in de code:
# 
# 1. Verhouding Toevoegen: Er wordt een nieuwe kolom Verhouding toegevoegd, die de verhouding berekent van overwinningen tot inkomende transfers voor elk team.
# 
# 2. Visualisatie: We gebruiken een staafgrafiek (bar chart) om de verhouding te tonen voor het geselecteerde team. Dit helpt om de impact van inkomende transfers te begrijpen in verhouding tot de behaalde overwinningen.
# 
# 3. Interactief Dropdown Menu: Het dropdownmenu laat de gebruiker een team selecteren, waarna de verhouding voor dat team wordt weergegeven.

# %% [markdown]
# Resultaat:
# 
# De visualisatie zal de verhouding van overwinningen per inkomende transfer voor elk geselecteerd team tonen. Dit kan een goed inzicht geven in hoe efficiënt teams zijn met hun inkomende transfers.

# %% [markdown]
# Waarom? 
# 
# De verhouding van inkomende transfers tot overwinningen helpt om de effectiviteit van het transferbeleid van een team te begrijpen. Het biedt een dieper inzicht dan de absolute aantallen, door de prestaties te relativeren naar het aantal transfers dat elk team heeft uitgevoerd. Dit stelt ons in staat om te zien welke teams het meest efficiënt zijn in het vertalen van hun transferinspanningen naar succes op het veld.
# 
# de analyse verdiept en context biedt bij de vorige visualisaties, waardoor je een completer beeld krijgt van hoe teams hun transferbeleid in de praktijk gebruiken en hoe dat hun prestaties beïnvloedt.

# %% [markdown]
# DROP DOWN MENU

# %%
import plotly.express as px
import pandas as pd
from ipywidgets import widgets, interact

# Voorbeelddata
df = pd.DataFrame({
    'Team': ['Arsenal', 'Chelsea', 'Liverpool', 'Tottenham', 'Man City', 'Manchester United', 'Leicester City', 'Everton', 'West Ham', 'Aston Villa'],
    'Inkomende_Transfers': [5, 7, 3, 8, 6, 4, 2, 3, 4, 5],
    'Aantal_Overwinningen': [20, 18, 22, 15, 23, 19, 14, 12, 16, 13]
})

# Voeg een nieuwe kolom toe voor de verhouding van overwinningen per inkomende transfer
df['Verhouding'] = df['Aantal_Overwinningen'] / df['Inkomende_Transfers']

# Functie die de grafiek bijwerkt op basis van het geselecteerde team
def update_graph(selected_team):
    # Filter de data op basis van het geselecteerde team
    filtered_df = df[df['Team'] == selected_team]
    
    # Maak de grafiek van de verhouding van overwinningen per inkomende transfer
    fig = px.bar(filtered_df, x='Team', y='Verhouding',
                 title=f"Verhouding van Overwinningen tot Inkomende Transfers voor {selected_team}",
                 labels={'Verhouding': 'Verhouding Overwinningen / Inkomende Transfers', 'Team': 'Team'},
                 color='Team', 
                 color_discrete_sequence=px.colors.qualitative.Set2)  # Kleurenschema consistent
    
    # Toon de grafiek
    fig.show()

# Dropdown menu voor team selectie
team_dropdown = widgets.Dropdown(
    options=df['Team'].tolist(),  # Maak een lijst van alle teams voor het dropdown-menu
    value='Arsenal',  # Standaard geselecteerd team
    description='Selecteer Team:',
)

# Maak de interactieve visualisatie met de dropdown
interact(update_graph, selected_team=team_dropdown)



# %%
import requests
import pandas as pd

# Voer je API-sleutel in hier
api_key = "703e38c5af704ea2b71e33878e34d5c4"

# De URL voor het ophalen van de ranglijst van de Premier League
url = "http://api.football-data.org/v4/competitions/2021/standings"

# Stel de headers in met de API-sleutel
headers = {
    "X-Auth-Token": api_key
}

# Verstuur een GET-verzoek naar de API
response = requests.get(url, headers=headers)

# Controleer of de aanvraag succesvol was (statuscode 200)
if response.status_code == 200:
    # Converteer het antwoord naar JSON-formaat
    data = response.json()

    # Inspecteer de data om te zien wat we moeten extraheren
    print("Volledige respons:", data)

    # Controleer of 'standings' aanwezig is in de respons
    if 'standings' in data:
        # De data zit in de eerste tabel van de standings (meestal de reguliere competitie)
        standings_table = data['standings'][0]['table']

        # Maak een lijst van dictionaries met relevante teaminformatie
        standings_data = []
        for team in standings_table:
            standings_data.append({
                'team': team['team']['name'],
                'position': team['position'],
                'playedGames': team['playedGames'],
                'won': team['won'],
                'draw': team['draw'],
                'lost': team['lost'],
                'goalsFor': team['goalsFor'],
                'goalsAgainst': team['goalsAgainst'],
                'goalDifference': team['goalDifference'],
                'points': team['points']
            })

        # Zet de teamgegevens om in een pandas DataFrame
        df = pd.DataFrame(standings_data)

        # Sla het DataFrame op in een CSV-bestand
        df.to_csv('premier_league_standings.csv', index=False)

        print("De ranglijst van de Premier League is succesvol opgeslagen in 'premier_league_standings.csv'.")
    else:
        print("Er is geen ranglijst gevonden in de respons.")
else:
    print(f"Er ging iets mis. Statuscode: {response.status_code}")


# %%
import pandas as pd
from tabulate import tabulate

# Lees de opgeslagen CSV met standings
df = pd.read_csv("premier_league_standings.csv")

# Print een mooie tabel in de terminal
print(tabulate(df, headers='keys', tablefmt='fancy_grid'))


# %%
import pandas as pd

# Lees de dataset in
df = pd.read_csv("transfers.csv")

# Zet de 'transfer_date' kolom om naar datetime-formaat
df['transfer_date'] = pd.to_datetime(df['transfer_date'])

# Filter op transfers in 2025
df_2025 = df[df['transfer_date'].dt.year == 2025]

# Definieer de lijst van Premier League-clubs
premier_league_clubs = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Chelsea", 
    "Crystal Palace", "Everton", "Fulham", "Leeds United", "Leicester City", 
    "Liverpool", "Manchester City", "Manchester United", "Newcastle United", 
    "Nottingham Forest", "Southampton", "Tottenham Hotspur", "West Ham United", 
    "Wolverhampton Wanderers", "Ipswich Town"
]

# Filter op inkomende transfers (to_club_name moet in de lijst van Premier League-clubs staan)
df_incoming = df_2025[df_2025['to_club_name'].isin(premier_league_clubs)]

# Tel het aantal inkomende transfers per Premier League-club
incoming_transfers_per_club = df_incoming['to_club_name'].value_counts()

# Print het resultaat
print(incoming_transfers_per_club)


# %%
print(response.json())

# %%
import requests
import pandas as pd
from datetime import datetime

# Maak een GET-aanroep naar de API
url = "https://api.football-data.org/v4/competitions/PL/matches"
headers = {"X-Auth-Token": "703e38c5af704ea2b71e33878e34d5c4"}  # vervang 'jouw_api_sleutel' met je eigen sleutel
response = requests.get(url, headers=headers)

# Controleer of de aanvraag succesvol was
if response.status_code == 200:
    data = response.json()
else:
    print("Er is iets mis gegaan met het ophalen van de data.")
    exit()

# Verkrijg alle wedstrijden
matches = data['matches']

# Filter wedstrijden die in 2025 zijn gespeeld
matches_2025 = []
for match in matches:
    match_date = datetime.strptime(match['utcDate'], '%Y-%m-%dT%H:%M:%SZ')
    if match_date.year == 2025:
        matches_2025.append(match)


# %%
# Lijst van Premier League-teams
teams = [
    "Arsenal FC", "Chelsea FC", "Liverpool FC", "Tottenham Hotspur FC", "Manchester City FC", 
    "Manchester United FC", "Leicester City FC", "Everton FC", "West Ham United FC", 
    "Aston Villa FC", "Newcastle United FC", "Brighton & Hove Albion FC", "Brentford FC", 
    "Crystal Palace FC", "Wolverhampton Wanderers FC", "Southampton FC", "AFC Bournemouth", 
    "Nottingham Forest FC", "Fulham FC", "Ipswich Town FC"
]

# Maak een dictionary om de punten bij te houden
team_stats = {team: {"points": 0} for team in teams}

# Verwerk de wedstrijden uit 2025
for match in matches_2025:
    home_team = match['homeTeam']['name']
    away_team = match['awayTeam']['name']
    home_score = match['score']['fullTime']['home']
    away_score = match['score']['fullTime']['away']
    
    # Bijwerken van de punten
    if home_score > away_score:
        team_stats[home_team]["points"] += 3
    elif away_score > home_score:
        team_stats[away_team]["points"] += 3
    else:
        team_stats[home_team]["points"] += 1
        team_stats[away_team]["points"] += 1

# Zet de data om in een DataFrame
stand_data = []
for team, stats in team_stats.items():
    stand_data.append({
        "Team": team,
        "Points": stats["points"]
    })

stand_df = pd.DataFrame(stand_data)

# Sorteer de stand op basis van de punten
stand_df = stand_df.sort_values(by="Points", ascending=False).reset_index(drop=True)

# Bekijk de huidige stand
print(stand_df)


# %%
import requests
import pandas as pd
from datetime import datetime

# Maak een GET-aanroep naar de API
url = "https://api.football-data.org/v4/competitions/PL/matches"
headers = {"X-Auth-Token": "703e38c5af704ea2b71e33878e34d5c4"}  # vervang 'jouw_api_sleutel' met je eigen sleutel
response = requests.get(url, headers=headers)

# Controleer of de aanvraag succesvol was
if response.status_code == 200:
    data = response.json()
else:
    print("Er is iets mis gegaan met het ophalen van de data.")
    exit()

# Verkrijg alle wedstrijden
matches = data['matches']

# Filter wedstrijden die in 2025 zijn gespeeld
matches_2025 = []
for match in matches:
    match_date = datetime.strptime(match['utcDate'], '%Y-%m-%dT%H:%M:%SZ')
    if match_date.year == 2025:
        matches_2025.append(match)

# Lijst van Premier League-teams
teams = [
    "Arsenal FC", "Chelsea FC", "Liverpool FC", "Tottenham Hotspur FC", "Manchester City FC", 
    "Manchester United FC", "Leicester City FC", "Everton FC", "West Ham United FC", 
    "Aston Villa FC", "Newcastle United FC", "Brighton & Hove Albion FC", "Brentford FC", 
    "Crystal Palace FC", "Wolverhampton Wanderers FC", "Southampton FC", "AFC Bournemouth", 
    "Nottingham Forest FC", "Fulham FC", "Ipswich Town FC"
]

# Maak een dictionary om de punten bij te houden
team_stats = {team: {"points": 0} for team in teams}

# Verwerk de wedstrijden uit 2025
for match in matches_2025:
    home_team = match['homeTeam']['name']
    away_team = match['awayTeam']['name']
    home_score = match['score']['fullTime']['home']
    away_score = match['score']['fullTime']['away']
    
    # Controleer of de score beschikbaar is voor beide teams
    if home_score is not None and away_score is not None:
        # Bijwerken van de punten
        if home_score > away_score:
            team_stats[home_team]["points"] += 3
        elif away_score > home_score:
            team_stats[away_team]["points"] += 3
        else:
            team_stats[home_team]["points"] += 1
            team_stats[away_team]["points"] += 1

# Zet de data om in een DataFrame
stand_data = []
for team, stats in team_stats.items():
    stand_data.append({
        "Team": team,
        "Points": stats["points"]
    })

stand_df = pd.DataFrame(stand_data)

# Sorteer de stand op basis van de punten
stand_df = stand_df.sort_values(by="Points", ascending=False).reset_index(drop=True)

# Bekijk de huidige stand
print(stand_df)

# Opslaan van de stand in een CSV-bestand
stand_df.to_csv("premier_league_stand_2025_points.csv", index=False)


# %%
# Als incoming_transfers_per_club een Series is, zet het om naar een DataFrame
incoming_transfers_per_club = incoming_transfers_per_club.reset_index()
incoming_transfers_per_club.columns = ['to_club_name', 'transfers']  # Geef de juiste kolomnamen

# Inspecteren van de kolommen in de nieuwe DataFrame
print(incoming_transfers_per_club.columns)


# %%
import pandas as pd

# Mergen van de DataFrames op basis van de teamnaam zonder 'FC' in de teamnaam
stand_df['Team_short'] = stand_df['Team'].str.replace(" FC", "")  # Verwijder " FC" uit de teamnaam voor een kortere naam

# Samenvoegen van de twee DataFrames
merged_df = pd.merge(stand_df[['Team_short', 'Points']], 
                     incoming_transfers_per_club[['to_club_name', 'transfers']], 
                     left_on='Team_short', 
                     right_on='to_club_name', 
                     how='left')

# Nu hebben we een DataFrame met teamnamen, aantal punten en aantal transfers
final_df = merged_df[['Team_short', 'Points', 'transfers']]

# We tonen het resulterende DataFrame
print(final_df)


# %%

# Zet de 'transfers' kolom om naar gehele getallen (int)
merged_df['transfers'] = merged_df['transfers'].fillna(0).astype(int)

# Controleer het resultaat
print(merged_df[['Team_short', 'Points', 'transfers']])


# %%
# Hernoem de kolommen
merged_df = merged_df.rename(columns={
    'Team_short': 'Team', 
    'Points': 'Punten in 2025', 
    'transfers': 'Inkomende transfers'
})

# Controleer het resultaat
print(merged_df[['Team', 'Punten in 2025', 'Inkomende transfers']])


# %%
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Zet data om naar DataFrame
merged_df = pd.DataFrame(data)

# Streamlit titel
st.title("Punten in 2025 per Aantal Inkomende Transfers")

# Slider voor het aantal inkomende transfers
min_transfers = st.slider(
    "Kies het minimum aantal inkomende transfers", 
    min_value=0, 
    max_value=16, 
    value=0,  # Startwaarde
    step=1,
    help="Selecteer het minimum aantal inkomende transfers"
)

# Filter de DataFrame op basis van de sliderwaarde
filtered_df = merged_df[merged_df['Inkomende transfers'] >= min_transfers]

# Maak het staafdiagram
fig = go.Figure()
fig.add_trace(go.Bar(
    x=filtered_df['Team'],
    y=filtered_df['Punten in 2025'],
    marker=dict(color='skyblue'),
    text=filtered_df['Team'],
    textposition='outside'
))

# Toon de grafiek
st.plotly_chart(fig)

# Toon de gefilterde tabel
st.write(f"Gegevens van teams met minstens {min_transfers} inkomende transfers:")
st.dataframe(filtered_df[['Team', 'Punten in 2025', 'Inkomende transfers']])

# %%
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import interactive


# Functie die de grafiek bijwerkt op basis van de sliderwaarde
def update_graph(min_transfers):
    # Filter de DataFrame op basis van de sliderwaarde
    filtered_df = merged_df[merged_df['Inkomende transfers'] >= min_transfers]
    
    # Maak de figuur voor het staafdiagram
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=filtered_df['Team'],
        y=filtered_df['Punten in 2025'],
        marker=dict(color='skyblue'),
        text=filtered_df['Team'],
        textposition='outside'
    ))
    
    # Update de layout en toon de grafiek
    fig.update_layout(
        title=f"Punten in 2025 per Aantal Inkomende Transfers (Min {min_transfers} transfers)",
        xaxis_title="Team",
        yaxis_title="Punten in 2025",
        showlegend=False
    )
    fig.show()

# Maak de slider
slider = widgets.IntSlider(value=0, min=0, max=16, step=1, description='Min Transfers:')

# Maak een interactieve grafiek met de slider
interactive_plot = interactive(update_graph, min_transfers=slider)
interactive_plot


# %%
pip install streamlit plotly


# %%
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Functie voor het filteren van de DataFrame en het maken van de grafiek
@st.cache
def get_filtered_data(min_transfers):
    # Maak de 'merged_df' DataFrame (gebruik je eigen DataFrame hier)

    # Filter de DataFrame op basis van de sliderwaarde
    filtered_df = merged_df[merged_df['Inkomende transfers'] >= min_transfers]
    
    return filtered_df

# Streamlit titel
st.title("Punten in 2025 per Aantal Inkomende Transfers")

# Maak een slider in Streamlit voor het aantal inkomende transfers
min_transfers = st.slider(
    "Kies het minimum aantal inkomende transfers", 
    min_value=0, 
    max_value=16, 
    value=0,  # Startwaarde
    step=1,
    help="Selecteer het minimum aantal inkomende transfers"
)

# Filter de DataFrame op basis van de sliderwaarde
filtered_df = merged_df[merged_df['Inkomende transfers'] >= min_transfers]

# Maak de staafgrafiek
fig = go.Figure()
fig.add_trace(go.Bar(
    x=filtered_df['Team'],
    y=filtered_df['Punten in 2025'],
    marker=dict(color='skyblue'),
    text=filtered_df['Team'],
    textposition='outside'
))

# Update de layout van de grafiek
fig.update_layout(
    title=f"Punten in 2025 per Aantal Inkomende Transfers (Min {min_transfers} transfers)",
    xaxis_title="Team",
    yaxis_title="Punten in 2025",
    showlegend=False
)

# Toon de grafiek in Streamlit
st.plotly_chart(fig)

# Toon de gefilterde tabel in Streamlit
st.write(f"Gegevens van teams met minstens {min_transfers} inkomende transfers:")
st.dataframe(filtered_df[['Team', 'Punten in 2025', 'Inkomende transfers']])


# %%
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import interactive


# Functie die de grafiek bijwerkt op basis van de sliderwaarde
def update_graph(min_transfers):
    # Filter de DataFrame op basis van de sliderwaarde
    filtered_df = merged_df[merged_df['Inkomende transfers'] >= min_transfers]
    
    # Maak de figuur voor het staafdiagram
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=filtered_df['Team'],
        y=filtered_df['Punten in 2025'],
        marker=dict(color='skyblue'),
        text=filtered_df['Team'],
        textposition='outside'
    ))
    
    # Update de layout en toon de grafiek
    fig.update_layout(
        title=f"Punten in 2025 per Aantal Inkomende Transfers (Min {min_transfers} transfers)",
        xaxis_title="Team",
        yaxis_title="Punten in 2025",
        showlegend=False
    )
    fig.show()

# Maak de slider
slider = widgets.IntSlider(value=0, min=0, max=16, step=1, description='Min Transfers:')

# Maak een interactieve grafiek met de slider
interactive_plot = interactive(update_graph, min_transfers=slider)
interactive_plot



