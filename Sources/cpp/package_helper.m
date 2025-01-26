#import "package_helper.h"
#import <Foundation/Foundation.h>
#include <sys/utsname.h>

NSString *Get_Machine_Hardware_Name(void) {
    struct utsname sysinfo;
    int retVal = uname(&sysinfo);
    if (EXIT_SUCCESS != retVal) return nil;
    return [NSString stringWithUTF8String:sysinfo.machine];
}
