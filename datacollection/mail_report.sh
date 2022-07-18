#!/bin/bash

FROM='stealwatcher'
TO='idrisstafo9@gmail.com'
SUBJECT="Report for $1"
REPORT_FILE='readme.txt'
NUMBER_OF_PARAMS=$#

mail_report(){
  if [[ ${NUMBER_OF_PARAMS} == 2 ]]
  then
    REPORT_FILE="$2"
  fi
	if [[  -e ${REPORT_FILE}  ]]
	then
		cat ${REPORT_FILE} |mail -s "${SUBJECT}" "$TO" -a "From: ${FROM} -A ${REPORT_FILE}"
	else
		echo "Nothing to do"
	fi

}
mail_report "$@"