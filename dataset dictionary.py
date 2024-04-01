import numpy as np


dataset_dictionary = {
    'category': {
        0: 'hat',
        1: 'shirt',
        2: 'sweater',
        3: 'hoodie',
        4: 'coat',
        5: 'shorts',
        6: 'pants',
        7: 'skirt',
        8: 'socks',
        9: 'shoes',
        10: 'accessory', },

    'material': {
        0: 'cotton',
        1: 'wool',
        2: 'polyester',
        3: 'silk',
        4: 'linen',
        5: 'other' },
    
    'warmth': {
        0: 'not_applicable',
        1: 'for_hot_weather',
        2: 'for_warm_weather',
        3: 'versitile',
        4: 'for_cool_weather',
        5: 'for_cold_weather',
        6: 'for_arctic_weather' },

    'water_resistance': {
        0: 'not_applicable',
        1: 'no_resistance',
        2: 'mild_resistance',
        3: 'moderate_resistance',
        4: 'high_resistance',
        5: 'waterproof' },

    'silhouette': {
        0: 'not_applicable',
        1: 'skinny_fit',
        2: 'slim_fit',
        3: 'standard_fit',
        4: 'loose_fit',
        5: 'oversized_fit' }

}


dark_academia_color_palette = {
'#422d20', '#421220', '#5a3925,' '#81613b', '#774c3a',
'#271d20', '#b6aca8', '#2b271c', '#321f19', '#5a2827',
'#a8835d', '#241b24', '#5c343b', '#22151f', '#5b3630',
'#bdac86', '#54342b', '#3c3838', '#3d3232', '#202430',
'#8f5e5f', '#120f12', '#8d7458', '#704a2e', '#b69b7d',
'#2b2832', '#b89c8c', '#472f26', '#43473e', '#341f22'
}

def calculate_color_similarity(color, palette):
    # Example: Calculate color similarity using Euclidean distance
    distances = [np.linalg.norm(np.array(color) - np.array(palette_color)) for palette_color in palette]
    return np.mean(distances)

clothing_data['color_similarity_palette1'] = clothing_data['color'].apply(lambda x: calculate_color_similarity(x, preferred_palettes[0]))