<?php
    include 'local_paths.php';
    $frame = $_REQUEST["framename"];
    $labels = $_REQUEST["labels"];
    $nick = $_REQUEST["nick"];

    $script = "source/php_called_scripts/register_labels.py";
    $args = $labels." ".$frame." ".$nick." 2>&1";
    $cmd = $python_interpreter." ".$script_base.$script." ".$args;

    $out = system($cmd, $errorcode);

    if ($errorcode == 0){
        echo "OK";
    else{
        echo $out;
    }
?>