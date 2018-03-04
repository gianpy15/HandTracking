<?php
    include 'local_paths.php';
    $frame = $_POST["framename"];
    $labels = $_POST["labels"];
    $nick = $_POST["nick"];

    $script = "source/php_called_scripts/register_labels.py";
    $args = $labels." ".$frame." ".$nick." 2>&1";
    $cmd = $python_interpreter." ".$script_base.$script." ".$args;
    $out = exec($cmd, $errorcode);

    if ($errorcode == 0 && $out == ""){
        echo "OK";
    }else{
        echo $cmd."\n\n".$out;
    }
?>