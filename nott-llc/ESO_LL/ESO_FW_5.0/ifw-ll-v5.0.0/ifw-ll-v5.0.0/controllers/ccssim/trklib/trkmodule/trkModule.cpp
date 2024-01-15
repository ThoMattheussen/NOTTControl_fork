///////////////////////////////////////////////////////////////////////////////
// trkModule.cpp
#include "TcPch.h"
#pragma hdrstop

#include "trkModule.h"
#include "TimeFunctions.h"
#include "slalib/slalib.h"
#include "Computation.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif
DEFINE_THIS_FILE()

///////////////////////////////////////////////////////////////////////////////
// CtrkModule
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// Collection of interfaces implemented by module CtrkModule
BEGIN_INTERFACE_MAP(CtrkModule)
	INTERFACE_ENTRY_ITCOMOBJECT()
	INTERFACE_ENTRY(IID_ITcADI, ITcADI)
	INTERFACE_ENTRY(IID_ITcWatchSource, ITcWatchSource)
///<AutoGeneratedContent id="InterfaceMap">
	INTERFACE_ENTRY(IID_ITcCyclic, ITcCyclic)
///</AutoGeneratedContent>
END_INTERFACE_MAP()

IMPLEMENT_ITCOMOBJECT(CtrkModule)
IMPLEMENT_ITCOMOBJECT_SETSTATE_LOCKOP2(CtrkModule)
IMPLEMENT_ITCADI(CtrkModule)
IMPLEMENT_ITCWATCHSOURCE(CtrkModule)


///////////////////////////////////////////////////////////////////////////////
// Set parameters of CtrkModule 
BEGIN_SETOBJPARA_MAP(CtrkModule)
	SETOBJPARA_DATAAREA_MAP()
///<AutoGeneratedContent id="SetObjectParameterMap">
	SETOBJPARA_VALUE(PID_TcTraceLevel, m_TraceLevelMax)
	SETOBJPARA_VALUE(PID_trkModuleParameter, m_Parameter)
	SETOBJPARA_ITFPTR(PID_Ctx_TaskOid, m_spCyclicCaller)
///</AutoGeneratedContent>
END_SETOBJPARA_MAP()

///////////////////////////////////////////////////////////////////////////////
// Get parameters of CtrkModule 
BEGIN_GETOBJPARA_MAP(CtrkModule)
	GETOBJPARA_DATAAREA_MAP()
///<AutoGeneratedContent id="GetObjectParameterMap">
	GETOBJPARA_VALUE(PID_TcTraceLevel, m_TraceLevelMax)
	GETOBJPARA_VALUE(PID_trkModuleParameter, m_Parameter)
	GETOBJPARA_ITFPTR(PID_Ctx_TaskOid, m_spCyclicCaller)
///</AutoGeneratedContent>
END_GETOBJPARA_MAP()

///////////////////////////////////////////////////////////////////////////////
// Get watch entries of CtrkModule
BEGIN_OBJPARAWATCH_MAP(CtrkModule)
	OBJPARAWATCH_DATAAREA_MAP()
///<AutoGeneratedContent id="ObjectParameterWatchMap">
///</AutoGeneratedContent>
END_OBJPARAWATCH_MAP()

///////////////////////////////////////////////////////////////////////////////
// Get data area members of CtrkModule
BEGIN_OBJDATAAREA_MAP(CtrkModule)
///<AutoGeneratedContent id="ObjectDataAreaMap">
	OBJDATAAREA_VALUE(ADI_trkModuleInputs, m_Inputs)
	OBJDATAAREA_VALUE(ADI_trkModuleOutputs, m_Outputs)
///</AutoGeneratedContent>
END_OBJDATAAREA_MAP()


///////////////////////////////////////////////////////////////////////////////
CtrkModule::CtrkModule()
	: m_Trace(m_TraceLevelMax, m_spSrv)
	, m_TraceLevelMax(tlAlways)
	, m_counter(0)
{
///<AutoGeneratedContent id="MemberInitialization">
	m_TraceLevelMax = tlAlways;
	memset(&m_Parameter, 0, sizeof(m_Parameter));
	memset(&m_Inputs, 0, sizeof(m_Inputs));
	memset(&m_Outputs, 0, sizeof(m_Outputs));
///</AutoGeneratedContent>

}

///////////////////////////////////////////////////////////////////////////////
CtrkModule::~CtrkModule() 
{
}


///////////////////////////////////////////////////////////////////////////////
// State Transitions 
///////////////////////////////////////////////////////////////////////////////
IMPLEMENT_ITCOMOBJECT_SETOBJSTATE_IP_PI(CtrkModule)

///////////////////////////////////////////////////////////////////////////////
// State transition from PREOP to SAFEOP
//
// Initialize input parameters 
// Allocate memory
HRESULT CtrkModule::SetObjStatePS(PTComInitDataHdr pInitData)
{
	m_Trace.Log(tlVerbose, FENTERA);
	HRESULT hr = S_OK;
	IMPLEMENT_ITCOMOBJECT_EVALUATE_INITDATA(pInitData);

	// TODO: Add initialization code

	m_Trace.Log(tlVerbose, FLEAVEA "hr=0x%08x", hr);
	return hr;
}

///////////////////////////////////////////////////////////////////////////////
// State transition from SAFEOP to OP
//
// Register with other TwinCAT objects
HRESULT CtrkModule::SetObjStateSO()
{
	m_Trace.Log(tlVerbose, FENTERA);
	HRESULT hr = S_OK;

	
	// If following call is successful the CycleUpdate method will be called, 
	// possibly even before method has been left.
	hr = FAILED(hr) ? hr : AddModuleToCaller();

	// get reference to TC-RTime-Instance
	m_spRTime.SetOID(OID_TCRTIME_CTRL);
	hr = FAILED(hr) ? hr : m_spSrv->TcQuerySmartObjectInterface(m_spRTime);

	// Cleanup if transition failed at some stage
	if (FAILED(hr))
	{
		RemoveModuleFromCaller();
		m_spRTime = NULL;
	}

	

	m_Trace.Log(tlVerbose, FLEAVEA "hr=0x%08x", hr);
	return hr;
}

///////////////////////////////////////////////////////////////////////////////
// State transition from OP to SAFEOP
HRESULT CtrkModule::SetObjStateOS()
{
	m_Trace.Log(tlVerbose, FENTERA);

	HRESULT hr = S_OK;

	RemoveModuleFromCaller(); 
	
	m_spRTime = NULL;
	// TODO: Add any additional deinitialization

	m_Trace.Log(tlVerbose, FLEAVEA "hr=0x%08x", hr);
	return hr;
}

///////////////////////////////////////////////////////////////////////////////
// State transition from SAFEOP to PREOP
HRESULT CtrkModule::SetObjStateSP()
{
	HRESULT hr = S_OK;
	m_Trace.Log(tlVerbose, FENTERA);

	// TODO: Add deinitialization code

	m_Trace.Log(tlVerbose, FLEAVEA "hr=0x%08x", hr);
	return hr;
}

///<AutoGeneratedContent id="ImplementationOf_ITcCyclic">
HRESULT CtrkModule::CycleUpdate(ITcTask* ipTask, ITcUnknown* ipCaller, ULONG_PTR context)
{
	HRESULT hr = S_OK;
	long long dc_time = 0;
	ccsTIMEVAL utc;
	
	// Distributed clock timestamp at "begin of task" (independent from read time within task cycle)
	hr = FAILED(hr) ? hr : m_spRTime->GetCurDcTime(GETCURDCTIME_ACTUAL, &dc_time);

	timeGetAbsoluteDcTime(m_Inputs.TimeInfo.offset, &dc_time);
	timeGetUTCInFuture(dc_time, &utc, TIME_AHEAD);
	//timeGetUTC(dc_time, &utc);
	//m_Outputs.CcsDeterministic.time_tai = timeGetMudpiTime(dc_time, m_Inputs.TimeInfo.leap_second);
	m_Outputs.CcsDeterministic.time_tai = timeGetMudpiTimeInFuture(dc_time, m_Inputs.TimeInfo.leap_second, 0.05);
	m_Outputs.CcsDeterministic.time_utc = (double)(utc.tv_sec + utc.tv_usec * 0.000001);


	// do the full transformations
	ComputeTracking(
		m_Inputs.SlaParams,
		m_Inputs.MeanCoordinates,
		utc,
		m_Outputs.CcsDeterministic);
	
	return hr;
}
///</AutoGeneratedContent>

///////////////////////////////////////////////////////////////////////////////
HRESULT CtrkModule::AddModuleToCaller()
{
	m_Trace.Log(tlVerbose, FENTERA);

	HRESULT hr = S_OK;
	if ( m_spCyclicCaller.HasOID() )
	{
		if ( SUCCEEDED_DBG(hr = m_spSrv->TcQuerySmartObjectInterface(m_spCyclicCaller)) )
		{
			if ( FAILED(hr = m_spCyclicCaller->AddModule(m_spCyclicCaller, THIS_CAST(ITcCyclic))) )
			{
				m_spCyclicCaller = NULL;
			}
		}
	}
	else
	{
		hr = ADS_E_INVALIDOBJID; 
		SUCCEEDED_DBGT(hr, "Invalid OID specified for caller task");
	}

	m_Trace.Log(tlVerbose, FLEAVEA "hr=0x%08x", hr);
	return hr;
}

///////////////////////////////////////////////////////////////////////////////
VOID CtrkModule::RemoveModuleFromCaller()
{
	m_Trace.Log(tlVerbose, FENTERA);

	if ( m_spCyclicCaller )
	{
		m_spCyclicCaller->RemoveModule(m_spCyclicCaller);
	}
	m_spCyclicCaller	= NULL;

	m_Trace.Log(tlVerbose, FLEAVEA);
}

