from minimalyolo.utils import rescale_boxes
coords = [0.525132893041237, 0.6454825315005728, 0.11030927835051534, 0.15945017182130575]
coords = rescale_boxes(coords, imsize)