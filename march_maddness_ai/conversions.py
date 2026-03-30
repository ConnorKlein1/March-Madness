from copy import copy

conversions = {
    "FDU": "Fairleigh Dickinson",
    "NC State" : "N.C. State" ,
    "UConn": "Connecticut",
    "McNeese": "McNeese St.",
    "Miami (FL)": "Miami FL",
    "Miami (Fla)": "Miami FL",
    "Miami (Fla.)": "Miami FL",
    "Texas A&M - CC": "Texas A&M Corpus Chris",
    "Texas A&M-CC": "Texas A&M Corpus Chris",
    "Tex. A&M CC": "Texas A&M Corpus Chris",
    "North Kentucky": "Northern Kentucky",
    "Col of Charleston": "Charleston",
    "College of Charleston": "Charleston",
    "UCSB": "UC Santa Barbara",
    "S. Dakota St.": "South Dakota St.",
    "St. Peter’s": "Saint Peter's",
    "NM St.": "New Mexico St.",
    "Mt. St. Mary's": "Mount State Mary's",
    "St. Mary’s" : "Saint Mary's",
    "Saint Mary’s" : "Saint Mary's",
    "Eastern Wash.": "Eastern Washington",
    "Loyola Chi.": "Loyola Chicago",
    "N.C. Central": "North Carolina Central",
    "Pennsylvania": "Penn",
    "Southern California": "USC",
    "Hawai'i": "Hawaii",
    "Ole Miss": "Mississippi",
    "Arkansas-Pine Bluff": "Arkansas Pine Bluff",
    "St. Joseph's": "Saint Joseph's",
    "Pitt" : "Pittsburgh",
    "SE Missouri St." : "Southeast Missouri St.",
    "FAU" : "Florida Atlantic",
    "Gardner-Webb" : "Gardner Webb",
    "UMass" : "Massachusetts",
    "Loyola (Md.)" : "Loyola MD",
    "Detroit" : "Detroit Mercy",
    "App. St.":"Appalachian St.",
    "Arkansas-Little Rock":"Little Rock",
    "Long Island" : "LIU",
    "Omaha" : "Nebraska Omaha",
    # "Texas-Arlington" : 
}

def try_similar_names(df, teamName):
    og_name = copy(teamName)
    team = df.loc[df['Team'].str.strip() == teamName]
    if teamName == "Long Island" or teamName == "Texas-Arlington":
        test = 1
    # conversion list
    if team.empty and teamName in conversions:
        teamName = conversions[teamName]
        team = df.loc[df['Team'].str.strip() == teamName]
        teamName = copy(og_name)
    # conversion list + st -> state
    if team.empty and "St." in teamName and teamName in conversions:
        teamName = conversions[teamName]
        teamName = teamName.replace("St.", "State")
        team = df.loc[df['Team'].str.strip() == teamName]
        teamName = copy(og_name)
        # conversion list + st <- state
    if team.empty and "St." in teamName and teamName in conversions:
        teamName = conversions[teamName]
        teamName = teamName.replace("State", "St.")
        team = df.loc[df['Team'].str.strip() == teamName]
        teamName = copy(og_name)
    # st->state alone
    if team.empty and "St." in teamName:
        teamName = teamName.replace("St.", "State")
        team = df.loc[df['Team'].str.strip() == teamName]
        teamName = copy(og_name)
    # state->state
    if team.empty and "State" in teamName:
        teamName = teamName.replace("State", "St.")
        team = df.loc[df['Team'].str.strip() == teamName]
        teamName = copy(og_name)
    # state->state + conversions
    if team.empty and "State" in teamName in teamName and teamName in conversions:
        teamName = conversions[teamName]
        teamName = teamName.replace("State", "St.")
        team = df.loc[df['Team'].str.strip() == teamName]
        teamName = copy(og_name)
        
    if team.empty:
        print(f"{og_name} not found")

    return team