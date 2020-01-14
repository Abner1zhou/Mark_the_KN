import os
import pathlib

root = pathlib.Path(os.path.abspath(__file__)).parent
data_path = os.path.join(root, 'data')

History_path = os.path.join(data_path, '高中_地理', 'origin')
Geography_path = os.path.join(data_path, '高中_地理', 'origin')
Politics_path = os.path.join(data_path, '高中_政治', 'origin')
Biological_path = os.path.join(data_path, '高中_生物', 'origin')
# 我真的翻译的好累……
ancient_his_path = os.path.join(History_path, '古代史.csv')
modern_his_path = os.path.join(History_path, '近代史.csv')
contemporary_his_path = os.path.join(History_path, '现代史.csv')

population_and_cities_path = os.path.join(Geography_path, '人口与城市.csv')
development_path = os.path.join(Geography_path, '区域可持续发展.csv')
earth_path = os.path.join(Geography_path, '地球与地图.csv')
earth_in_universe_path = os.path.join(Geography_path, '宇宙中的地球.csv')
production_activities_path = os.path.join(Geography_path, '生产活动与地域联系.csv')

civic_morality_path =  os.path.join(Politics_path, '公民道德与伦理常识.csv')
current_politics_path = os.path.join(Politics_path, '时事政治.csv')
legal_path = os.path.join(Politics_path, '生活中的法律常识.csv')
scientific_thinking_path = os.path.join(Politics_path, '科学思维常识.csv')
scientific_socialism_path = os.path.join(Politics_path, '科学社会主义常识.csv')
economics_path = os.path.join(Politics_path, '经济学常识.csv')

cells_path = os.path.join(Biological_path, '分子与细胞.csv')
modern_biological_path = os.path.join(Biological_path, '现代生物技术专题.csv')
biotechnology_practice_path = os.path.join(Biological_path, '生物技术实践.csv')
biological_science_path = os.path.join(Biological_path, '生物科学与社会.csv')
steady_state_path = os.path.join(Biological_path, '稳态与环境.csv')
genetics_and_evolution = os.path.join(Biological_path, '遗传与进化.csv')
