<?php
    include 'local_paths.php';
    $frame = $_POST["framename"];

    $script = "source/php_called_scripts/uncache_frame.py";
    $args = $frame." 2>&1";
    $cmd = $python_interpreter." ".$script_base.$script." ".$args;
    $out = exec($cmd, $rets, $errorcode);

    if ($errorcode == 0 && $out == ""){
        echo "OK";
    }else{
        echo $cmd."\n\n".$out;
    }
?>