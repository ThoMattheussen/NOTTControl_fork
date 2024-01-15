///////////////////////////////////////////////////////////////////////////////
// trkmoduleClassFactory.cpp
#include "TcPch.h"
#pragma hdrstop

#include "trkmoduleClassFactory.h"
#include "trkmoduleServices.h"
#include "trkmoduleVersion.h"
#include "trkModule.h"

BEGIN_CLASS_MAP(CtrkmoduleClassFactory)
///<AutoGeneratedContent id="ClassMap">
	CLASS_ENTRY(CID_trkmoduleCtrkModule, SRVNAME_TRKMODULE "!CtrkModule", CtrkModule)
///</AutoGeneratedContent>
END_CLASS_MAP()

CtrkmoduleClassFactory::CtrkmoduleClassFactory() : CObjClassFactory()
{
	TcDbgUnitSetImageName(TCDBG_UNIT_IMAGE_NAME(SRVNAME_TRKMODULE)); 
#if defined(TCDBG_UNIT_VERSION)
	TcDbgUnitSetVersion(TCDBG_UNIT_VERSION(trkmodule));
#endif //defined(TCDBG_UNIT_VERSION)
}

