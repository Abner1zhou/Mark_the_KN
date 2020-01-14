# -*- coding: UTF-8 –*-
"""
@author: 周世聪
@contact: abner1zhou@gmail.com
@desc: 文件路径
"""
import os
import pathlib

root = pathlib.Path(os.path.abspath(__file__)).parent.parent
data_path = os.path.join(root, 'data', '百度题库')

History_path = os.path.join(data_path, '高中_历史', 'origin')
Geography_path = os.path.join(data_path, '高中_地理', 'origin')
Politics_path = os.path.join(data_path, '高中_政治', 'origin')
Biological_path = os.path.join(data_path, '高中_生物', 'origin')
# 我真的翻译的好累……
ancient_his_path = os.path.join(History_path, '古代史.csv')
modern_his_path = os.path.join(History_path, '近代史.csv')
contemporary_his_path = os.path.join(History_path, '现代史.csv')
history_path_all = [ancient_his_path,
                    modern_his_path,
                    contemporary_his_path
                    ]
history_section = ['古代史', '近代史', '现代史']

population_and_cities_path = os.path.join(Geography_path, '人口与城市.csv')
development_path = os.path.join(Geography_path, '区域可持续发展.csv')
earth_path = os.path.join(Geography_path, '地球与地图.csv')
earth_in_universe_path = os.path.join(Geography_path, '宇宙中的地球.csv')
production_activities_path = os.path.join(Geography_path, '生产活动与地域联系.csv')
geography_path_all = [population_and_cities_path,
                      development_path,
                      earth_path,
                      earth_in_universe_path,
                      production_activities_path
                      ]
geography_section = ['人口与城市', '区域可持续发展', '地球与地图', '宇宙中的地球', '生产活动与地域联系']

civic_morality_path = os.path.join(Politics_path, '公民道德与伦理常识.csv')
current_politics_path = os.path.join(Politics_path, '时事政治.csv')
legal_path = os.path.join(Politics_path, '生活中的法律常识.csv')
scientific_thinking_path = os.path.join(Politics_path, '科学思维常识.csv')
scientific_socialism_path = os.path.join(Politics_path, '科学社会主义常识.csv')
economics_path = os.path.join(Politics_path, '经济学常识.csv')
politics_path_all = [civic_morality_path,
                     current_politics_path,
                     legal_path,
                     scientific_thinking_path,
                     scientific_socialism_path,
                     economics_path
                     ]
politics_section = ['公民道德与伦理常识', '时事政治', '生活中的法律常识',
                    '科学思维常识', '科学社会主义常识', '经济学常识']

cells_path = os.path.join(Biological_path, '分子与细胞.csv')
modern_biological_path = os.path.join(Biological_path, '现代生物技术专题.csv')
biotechnology_practice_path = os.path.join(Biological_path, '生物技术实践.csv')
biological_science_path = os.path.join(Biological_path, '生物科学与社会.csv')
steady_state_path = os.path.join(Biological_path, '稳态与环境.csv')
genetics_and_evolution = os.path.join(Biological_path, '遗传与进化.csv')
biological_path_all = [cells_path,
                       modern_biological_path,
                       biotechnology_practice_path,
                       biological_science_path,
                       genetics_and_evolution
                       ]
biological_section = ['分子与细胞', '现代生物技术专题', '生物技术实践', '生物科学与社会', '稳态与环境', '遗传与进化']
