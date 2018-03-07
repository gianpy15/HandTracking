
INFO_FORMAT = {"color": GREEN, "size": "3"};
NOTICE_FORMAT = {"color": BLUE, "size": "3"};
WARNING_FORMAT = {"color": RED, "size": "4"};


class TextFormatter{
    constructor(target){
        this.target_element = target;
        this.current_text = '';
        this.resetText();
    }

    addNotice(text){
        this.current_text += applyFont(text, NOTICE_FORMAT) + "<br>";
        this.refresh();
    }

    addWarning(text){
        this.current_text += applyFont(text, WARNING_FORMAT) + "<br>";
        this.refresh();
    }

    resetText(){
        this.current_text = this.buildTextHeader();
        this.refresh();
    }

    refresh(){
        this.target_element.innerHTML = this.current_text;
    }

    buildTextHeader(){
        let ret = "";
        if(nickname != ''){
            ret += applyFont("Nickname: "+nickname, INFO_FORMAT) + "<br>";
        }
        ret += applyFont("Frame: " + extractFrameName(target_img_url), INFO_FORMAT) + "<br>";
        ret += applyFont("Video completion: ", INFO_FORMAT);
        ret += applyFont((completion_rate * 100).toFixed(2) + "%",
            {"color": getScalingColor(completion_rate), "size": "3"}) + "<br>";
        return ret;
    }


}

function applyFont(text, formatInfo){
    let ret = "<font";
    if(formatInfo["color"] != null)
        ret += " color="+formatInfo["color"];
    if(formatInfo["size"] != null)
        ret += " size="+formatInfo["size"];
    if(formatInfo["face"] != null)
        ret += " face="+formatInfo["face"];
    ret += ">"+text+"</font>";
    return ret;
}

function getScalingColor(rate){
    let green = Math.round(rate * 255);
    let red = Math.round((1-rate) * 255);
    let redstr = ("0" + red.toString(16)).substr(-2);
    let greenstr = ("0" + green.toString(16)).substr(-2);
    return "#"+redstr+greenstr+"00";
}
