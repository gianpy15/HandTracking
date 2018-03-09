var PALMCOL = '#000000';

var THUMBCOL1 = '#285E1C';
var THUMBCOL2 = '#388E3C';
var THUMBCOL3 = '#58AE4C';

var INDEXCOL1 = '#95AF21';
var INDEXCOL2 = '#DEEF41';
var INDEXCOL3 = '#FEFF71';

var MIDDLECOL1 = '#750000';
var MIDDLECOL2 = '#A50000';
var MIDDLECOL3 = '#F50000';

var RINGCOL1 = '#4B0072';
var RINGCOL2 = '#7B1FA2';
var RINGCOL3 = '#9B2FFF';

var BABYCOL1 = '#003560';
var BABYCOL2 = '#1565C0';
var BABYCOL3 = '#3585F0';


var SEGMENTS_LIST =
    {
        // WRIST
        0: [],
        //THUMB
        1: [[0, PALMCOL]],
        2: [[1, THUMBCOL1]],
        3: [[2, THUMBCOL2]],
        4: [[3, THUMBCOL3]],
        //INDEX
        5: [[0, PALMCOL], [1, PALMCOL]],
        6: [[5, INDEXCOL1]],
        7: [[6, INDEXCOL2]],
        8: [[7, INDEXCOL3]],
        //MIDDLE
        9: [[0, PALMCOL], [5, PALMCOL]],
        10: [[9, MIDDLECOL1]],
        11: [[10, MIDDLECOL2]],
        12: [[11, MIDDLECOL3]],
        //RING
        13: [[0, PALMCOL], [9, PALMCOL]],
        14: [[13, RINGCOL1]],
        15: [[14, RINGCOL2]],
        16: [[15, RINGCOL3]],
        //BABY
        17: [[0, PALMCOL], [13, PALMCOL]],
        18: [[17, BABYCOL1]],
        19: [[18, BABYCOL2]],
        20: [[19, BABYCOL3]],
    };
